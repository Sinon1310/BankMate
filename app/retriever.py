
from pathlib import Path

from langchain_chroma import Chroma
from app.embeddings import get_embedding_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"


def get_vector_store():
    embeddings = get_embedding_model()

    return Chroma(
        collection_name="bankmate_policies",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )


def get_retriever(search_type="similarity"):
    vector_store = get_vector_store()

    if search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 7,
                "lambda_mult": 0.5
            }
        )

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )


def search_with_scores(question, k=3):
    vector_store = get_vector_store()

    results = vector_store.similarity_search_with_score(
        question,
        k=k
    )

    return results

def search_relevant_documents(question, k=3, max_distance=1.3):
    """
    Retrieve only documents within the configured
    maximum distance.

    Lower distance indicates greater similarity.
    """

    results = search_with_scores(question, k)

    relevant_documents = [
        (document, distance)
        for document, distance in results
        if distance <= max_distance
    ]

    return relevant_documents




def search_by_policy(question, policy_id, k=3, section=None):

    vector_store = get_vector_store()

    metadata_filter = {
        "policy_id": policy_id
    }

    if section:
        metadata_filter = {
            "$and": [
                {"policy_id": policy_id},
                {"section": section.upper()}
            ]
        }

    results = vector_store.similarity_search(
        query=question,
        k=k,
        filter=metadata_filter
    )

    return results



if __name__ == "__main__":

    test_questions = [
        "What should he do if he notices an unauthorized transaction?",
        "What should a customer do about an unauthorized card transaction?",
        "What should a customer do if they notice an unauthorized card transaction?",
        "What is the bank's home loan interest rate?"
    ]

    for question in test_questions:

        print("\n" + "=" * 65)
        print("QUESTION:", question)
        print("=" * 65)

        # IMPORTANT: Raw distances, without threshold filtering
        results = search_with_scores(question, k=3)

        for index, (document, distance) in enumerate(results, start=1):

            print(f"\nRESULT {index}")
            print(f"Distance: {distance:.4f}")
            print("Source:", document.metadata.get("source"))
            print("Content:", document.page_content[:300])


    print("\nSECTION FILTERING TEST")

    results = search_by_policy(
        question="What happens when a refund is approved?",
        policy_id="POL-002",
        section="PROCESSING",
        k=3
    )

    for doc in results:
        print("\nPolicy:", doc.metadata["policy_id"])
        print("Section:", doc.metadata["section"])
        print("Content:", doc.page_content)
