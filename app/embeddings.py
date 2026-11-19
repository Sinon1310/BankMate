
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NORMALIZE_EMBEDDINGS = True

# Identifies the vectors an index was built with; part of its fingerprint.
EMBEDDING_ID = f"{EMBEDDING_MODEL_NAME}|normalized={NORMALIZE_EMBEDDINGS}"


# Loading the model takes ~3s and every search needs it. It is stateless
# after loading, so one instance per process is reused (index builds and
# searches share it).
@lru_cache(maxsize=1)
def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS}
    )

    return embeddings


if __name__ == "__main__":
    embeddings = get_embedding_model()

    sentences = [
        "How do I report an unauthorized transaction?",
        "What should I do if someone uses my card without permission?",
        "How can I request a refund?"
    ]

    vectors = embeddings.embed_documents(sentences)

    print("Number of sentences:", len(sentences))
    print("Vector dimensions:", len(vectors[0]))
    print("First 10 values:", vectors[0][:10])
