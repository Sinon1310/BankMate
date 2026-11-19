"""
Policy vector index lifecycle (ChromaDB).

`ensure_policy_index()` guarantees a complete, validated index before any
policy search:

- **Reuse**: a collection marked complete whose fingerprint (chunk IDs,
  chunk text, metadata, embedding model, index format) matches the
  current policy files, with the expected chunk count and metadata, is
  reused as-is. Nothing is rebuilt or overwritten.
- **Build**: if the index is missing, empty, incomplete, outdated or
  invalid, it is built into a temporary collection, validated, and only
  then swapped into place (the old collection is removed at the very
  last step). An interrupted build therefore leaves either the previous
  index or no index — never a half-filled one that looks valid.
- **Fail loudly**: any problem raises PolicyIndexError. Callers must
  never treat an unavailable index as "no relevant policy found".
- **Once per process**: the result is cached per directory and guarded
  by a lock, so repeated or concurrent initialisation builds at most once.

Run `python -m app.vector_store` to force a rebuild.
"""

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma

from app.document_loader import load_policy_documents, split_documents
from app.embeddings import EMBEDDING_ID, get_embedding_model


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "bankmate_policies"
BUILD_COLLECTION_NAME = COLLECTION_NAME + "__building"

# Bump when chunking or stored metadata changes, to force a rebuild.
INDEX_FORMAT_VERSION = "2"

REQUIRED_METADATA = ("policy_id", "policy_name", "version", "section", "source_file", "chunk_id")
POLICY_ID = re.compile(r"^POL-\d+$")

STATE_KEY = "bankmate_state"
STATE_COMPLETE = "complete"
STATE_BUILDING = "building"


class PolicyIndexError(RuntimeError):
    """The policy index is unavailable or could not be built."""


@dataclass
class IndexStatus:
    state: str          # missing | empty | incomplete | outdated | invalid | valid
    reason: str = ""
    count: int = 0
    built: bool = False


# ---------- Clients ----------

_clients = {}
_clients_lock = threading.Lock()


def get_client(persist_dir=None):
    """One Chroma client per directory, shared by builds and searches."""

    path = str(Path(persist_dir or CHROMA_DIR))

    with _clients_lock:
        if path not in _clients:
            _clients[path] = chromadb.PersistentClient(
                path=path, settings=Settings(anonymized_telemetry=False)
            )
        return _clients[path]


def open_policy_store(persist_dir=None, embeddings=None):
    """LangChain store over the (already ensured) policy collection."""

    return Chroma(
        client=get_client(persist_dir),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings or get_embedding_model(),
    )


# ---------- Expected contents ----------

def expected_chunks(documents=None):
    documents = load_policy_documents() if documents is None else documents
    return split_documents(documents)


def validate_chunks(chunks):
    """Raise PolicyIndexError unless chunks are indexable as-is."""

    if not chunks:
        raise PolicyIndexError("No policy chunks were produced from knowledge/policies.")

    seen = set()

    for chunk in chunks:
        missing = [key for key in REQUIRED_METADATA if not str(chunk.metadata.get(key) or "").strip()]
        if missing:
            raise PolicyIndexError(
                f"Policy chunk is missing metadata {missing}: "
                f"{chunk.metadata.get('source_file') or chunk.metadata.get('source')}"
            )
        if not POLICY_ID.match(chunk.metadata["policy_id"]):
            raise PolicyIndexError(f"Invalid policy ID {chunk.metadata['policy_id']!r}")
        if chunk.metadata["chunk_id"] in seen:
            raise PolicyIndexError(f"Duplicate chunk ID {chunk.metadata['chunk_id']!r}")
        if not chunk.page_content.strip():
            raise PolicyIndexError(f"Empty chunk {chunk.metadata['chunk_id']!r}")
        seen.add(chunk.metadata["chunk_id"])


def _stored_metadata(chunk):
    return {key: chunk.metadata[key] for key in REQUIRED_METADATA}


def fingerprint(chunks, embedding_id=EMBEDDING_ID):
    digest = hashlib.sha256()
    digest.update(f"format={INDEX_FORMAT_VERSION}|embedding={embedding_id}".encode())

    for chunk in sorted(chunks, key=lambda c: c.metadata["chunk_id"]):
        record = json.dumps(
            {"text": chunk.page_content, "metadata": _stored_metadata(chunk)},
            sort_keys=True,
        )
        digest.update(record.encode())

    return digest.hexdigest()


# ---------- Inspection ----------

def _get_collection(client, name):
    try:
        return client.get_collection(name)
    except Exception:  # noqa: BLE001 - Chroma raises various "does not exist" types
        return None


def inspect_index(persist_dir=None, chunks=None, embedding_id=EMBEDDING_ID):
    """Classify the stored index against the expected chunks (read-only)."""

    chunks = expected_chunks() if chunks is None else chunks
    collection = _get_collection(get_client(persist_dir), COLLECTION_NAME)

    if collection is None:
        return IndexStatus("missing", "No policy collection exists yet.")

    count = collection.count()

    if count == 0:
        return IndexStatus("empty", "The policy collection has no chunks.")

    metadata = collection.metadata or {}

    if metadata.get(STATE_KEY) != STATE_COMPLETE:
        return IndexStatus("incomplete", "The collection was not marked complete.", count)

    if metadata.get("fingerprint") != fingerprint(chunks, embedding_id):
        return IndexStatus("outdated", "Policy files, chunking or embeddings changed.", count)

    if count != len(chunks):
        return IndexStatus("invalid", f"Expected {len(chunks)} chunks, found {count}.", count)

    stored = collection.get(include=["metadatas"])
    expected_ids = {chunk.metadata["chunk_id"] for chunk in chunks}

    if set(stored["ids"]) != expected_ids:
        return IndexStatus("invalid", "Stored chunk IDs do not match the policy files.", count)

    for item in stored["metadatas"]:
        item = item or {}
        if any(not str(item.get(key) or "").strip() for key in REQUIRED_METADATA) \
                or not POLICY_ID.match(str(item.get("policy_id"))):
            return IndexStatus("invalid", "A stored chunk has missing or invalid metadata.", count)

    return IndexStatus("valid", "", count)


# ---------- Build ----------

def build_index(persist_dir=None, chunks=None, embeddings=None, embedding_id=EMBEDDING_ID):
    """
    Build into a temporary collection, validate, then swap into place.
    Raises PolicyIndexError; on failure any existing index is untouched.
    """

    chunks = expected_chunks() if chunks is None else chunks
    validate_chunks(chunks)
    client = get_client(persist_dir)

    if _get_collection(client, BUILD_COLLECTION_NAME) is not None:
        client.delete_collection(BUILD_COLLECTION_NAME)   # leftover from an interrupted build

    try:
        store = Chroma(
            client=client,
            collection_name=BUILD_COLLECTION_NAME,
            embedding_function=embeddings or get_embedding_model(),
            collection_metadata={STATE_KEY: STATE_BUILDING},
        )
        store.add_documents(chunks, ids=[chunk.metadata["chunk_id"] for chunk in chunks])

        building = client.get_collection(BUILD_COLLECTION_NAME)
        if building.count() != len(chunks):
            raise PolicyIndexError(
                f"Indexed {building.count()} of {len(chunks)} chunks."
            )

        # Point of no return: replace the old collection with the new one.
        if _get_collection(client, COLLECTION_NAME) is not None:
            client.delete_collection(COLLECTION_NAME)

        building.modify(
            name=COLLECTION_NAME,
            metadata={
                STATE_KEY: STATE_COMPLETE,
                "fingerprint": fingerprint(chunks, embedding_id),
                "format_version": INDEX_FORMAT_VERSION,
                "embedding": embedding_id,
                "chunk_count": len(chunks),
                "built_at": int(time.time()),
            },
        )

    except Exception as error:
        try:
            if _get_collection(client, BUILD_COLLECTION_NAME) is not None:
                client.delete_collection(BUILD_COLLECTION_NAME)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            logger.exception("Could not remove the temporary index collection")

        if isinstance(error, PolicyIndexError):
            raise
        raise PolicyIndexError(f"Building the policy index failed: {error}") from error

    logger.info("Built policy index: %d chunks", len(chunks))


# ---------- Ensure (startup / before search) ----------

_ready = {}
_ensure_lock = threading.Lock()


def ensure_policy_index(persist_dir=None, embeddings=None, documents=None,
                        embedding_id=EMBEDDING_ID, force=False):
    """
    Make sure a valid index exists, building it only when needed.
    Returns an IndexStatus; raises PolicyIndexError if no valid index can
    be provided. Cheap after the first successful call in a process.
    """

    path = str(Path(persist_dir or CHROMA_DIR))

    if not force and path in _ready:
        return _ready[path]

    with _ensure_lock:
        if not force and path in _ready:
            return _ready[path]

        try:
            chunks = expected_chunks(documents)
            validate_chunks(chunks)
        except PolicyIndexError:
            raise
        except Exception as error:
            raise PolicyIndexError(f"Policy documents could not be loaded: {error}") from error

        status = inspect_index(path, chunks, embedding_id)

        if status.state != "valid" or force:
            logger.warning("Policy index %s (%s); building it now.", status.state, status.reason)
            build_index(path, chunks, embeddings, embedding_id)
            status = inspect_index(path, chunks, embedding_id)

            if status.state != "valid":
                raise PolicyIndexError(
                    f"The rebuilt policy index failed validation: {status.reason}"
                )
            status.built = True

        _ready[path] = status
        return status


def is_index_ready(persist_dir=None):
    return str(Path(persist_dir or CHROMA_DIR)) in _ready


def reset_index_state():
    """Forget cached readiness and clients (tests and forced rebuilds)."""

    with _ensure_lock:
        _ready.clear()
    with _clients_lock:
        _clients.clear()


def create_vector_store():
    """Force a rebuild of the local index (`python -m app.vector_store`)."""

    status = ensure_policy_index(force=True)
    print(f"Policy index ready: {status.count} chunks in {CHROMA_DIR} "
          f"(collection {COLLECTION_NAME}).")
    return open_policy_store()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_vector_store()
