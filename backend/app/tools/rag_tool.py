from langchain.tools import tool
from backend.app.rag.vector_store import retrieve_knowledge

@tool
def search_veterinary_knowledge(query: str) -> list:
    """
    Searches the veterinary knowledge base for information related to cattle diseases, symptoms, and treatments.
    Args:
        query (str): The search query.
    Returns:
        list: A list of relevant document excerpts.
    """
    try:
        return retrieve_knowledge(query)
    except Exception as e:
        return [f"Error retrieving knowledge: {str(e)}"]
