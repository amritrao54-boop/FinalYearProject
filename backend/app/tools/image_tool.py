from langchain.tools import tool
from backend.app.models.image_model import analyze_cattle_image

@tool
def image_prediction_tool(image_path: str) -> dict:
    """
    Analyzes a cattle image and predicts the disease.
    Args:
        image_path (str): The absolute or relative path to the image file.
    Returns:
        dict: The prediction result containing 'prediction', 'confidence', and 'model'.
    """
    try:
        return analyze_cattle_image(image_path)
    except Exception as e:
        return {"error": str(e)}
