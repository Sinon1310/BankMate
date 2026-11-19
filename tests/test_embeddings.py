"""
The embedding model is loaded once per process and reused (latency fix).
Uses a fake model class, so no model download or load happens here.
"""

import app.embeddings as embeddings


class FakeEmbeddings:
    created = 0

    def __init__(self, **kwargs):
        FakeEmbeddings.created += 1
        self.kwargs = kwargs


def test_embedding_model_is_created_once_and_reused(monkeypatch):
    monkeypatch.setattr(embeddings, "HuggingFaceEmbeddings", FakeEmbeddings)
    embeddings.get_embedding_model.cache_clear()
    FakeEmbeddings.created = 0

    try:
        first = embeddings.get_embedding_model()
        second = embeddings.get_embedding_model()

        assert first is second
        assert FakeEmbeddings.created == 1
        # Same configuration as before the cache was added.
        assert first.kwargs == {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "model_kwargs": {"device": "cpu"},
            "encode_kwargs": {"normalize_embeddings": True},
        }
    finally:
        # Don't leak the fake into other tests.
        embeddings.get_embedding_model.cache_clear()
