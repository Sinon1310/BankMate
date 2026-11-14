
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
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
