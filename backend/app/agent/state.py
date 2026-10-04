from typing import TypedDict, List, Optional, Any
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    user_query: str
    animal_id: Optional[str]
    image_path: Optional[str]
    symptoms: List[str]
    
    # Tool outputs
    image_prediction: Optional[dict]
    symptom_prediction: Optional[dict]
    retrieved_documents: List[str]
    
    # Final results
    final_assessment: Optional[dict]
    
    # LangGraph messages
    messages: List[BaseMessage]
