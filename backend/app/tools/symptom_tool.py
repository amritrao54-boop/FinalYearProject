from langchain.tools import tool
from backend.app.models.symptom_model import predict_disease_from_symptoms

@tool
def symptom_prediction_tool(symptoms: list) -> dict:
    """
    Analyzes a list of cattle symptoms and predicts the disease using Machine Learning models.
    Args:
        symptoms (list): A list of symptoms (e.g., ['fever', 'loss_of_appetite']).
    Returns:
        dict: The prediction result containing 'prediction', 'probabilities', and 'model'.
    """
    try:
        return predict_disease_from_symptoms(symptoms)
    except Exception as e:
        return {"error": str(e)}
