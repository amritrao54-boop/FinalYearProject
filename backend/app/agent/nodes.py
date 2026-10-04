import os
import re
from backend.app.agent.state import AgentState
from backend.app.tools.image_tool import image_prediction_tool
from backend.app.tools.symptom_tool import symptom_prediction_tool
from backend.app.tools.rag_tool import search_veterinary_knowledge
from backend.app.models.symptom_model import SYMPTOMS_LIST

# Mapping common user keywords to model symptoms
SYNONYMS = {
    'loose motion': 'diarrhoea',
    'diarrhea': 'diarrhoea',
    'diarrhoea': 'diarrhoea',
    'fever': 'fever',
    'high temperature': 'fever',
    'cough': 'coughing',
    'coughing': 'coughing',
    'loss of appetite': 'loss_of_appetite',
    'not eating': 'loss_of_appetite',
    'reduced appetite': 'loss_of_appetite',
    'drooling': 'salivation',
    'salivation': 'salivation',
    'limp': 'lameness',
    'limping': 'lameness',
    'lameness': 'lameness',
    'swelling': 'swelling',
    'swollen': 'swelling',
    'depressed': 'depression',
    'depression': 'depression',
    'weight loss': 'weight_loss',
    'losing weight': 'weight_loss'
}

def extract_symptoms_from_text(text: str) -> list:
    if not text:
        return []
    text_lower = text.lower()
    found = set()
    for phrase, mapped_symptom in SYNONYMS.items():
        if phrase in text_lower:
            found.add(mapped_symptom)
    for sym in SYMPTOMS_LIST:
        if sym.replace('_', ' ') in text_lower or sym in text_lower:
            found.add(sym)
    return list(found)

def analyze_image_node(state: AgentState) -> dict:
    image_path = state.get("image_path")
    if image_path:
        print(f"Agent: Analyzing image {image_path}")
        result = image_prediction_tool.invoke({"image_path": image_path})
        return {"image_prediction": result}
    return {}

def analyze_symptoms_node(state: AgentState) -> dict:
    symptoms = state.get("symptoms", [])
    user_query = state.get("user_query", "")
    
    # If no explicit symptoms list provided, extract from query text
    if not symptoms and user_query:
        symptoms = extract_symptoms_from_text(user_query)
        
    if symptoms:
        print(f"Agent: Analyzing extracted/provided symptoms {symptoms}")
        result = symptom_prediction_tool.invoke({"symptoms": symptoms})
        return {"symptom_prediction": result, "symptoms": symptoms}
    return {}

def retrieve_knowledge_node(state: AgentState) -> dict:
    query = state.get("user_query", "")
    predictions = []
    
    img_pred = state.get("image_prediction")
    if img_pred and isinstance(img_pred, dict) and "prediction" in img_pred:
        predictions.append(img_pred["prediction"])
        
    sym_pred = state.get("symptom_prediction")
    if sym_pred and isinstance(sym_pred, dict) and "prediction" in sym_pred:
        predictions.append(sym_pred["prediction"])
        
    search_queries = [query] if query else []
    search_queries.extend([f"Symptoms and prevention of {p}" for p in predictions])
    
    docs = []
    if search_queries:
        combined_query = " ".join(search_queries)
        print(f"Agent: Retrieving knowledge for '{combined_query}'")
        try:
            docs = search_veterinary_knowledge.invoke({"query": combined_query})
        except Exception as e:
            print(f"RAG retrieval warning: {e}")
            docs = []
        
    return {"retrieved_documents": docs}

from pydantic import BaseModel, Field
from typing import List, Optional

class AssessmentOutput(BaseModel):
    predicted_condition: str = Field(description="The most likely condition based on the fused model predictions and symptoms.")
    confidence: str = Field(description="Confidence level or probability.")
    certainty_level: str = Field(description="E.g., 'preliminary', 'high', 'low'.")
    evidence: List[str] = Field(description="Supporting evidence from models and symptoms.")
    knowledge_sources: List[str] = Field(description="Excerpts or summaries from the retrieved veterinary knowledge.")
    recommendations: List[str] = Field(description="Recommended next steps and general safety guidance.")
    warning: str = Field(description="A warning that this is an AI-assisted preliminary assessment and should not replace veterinary examination.")

def generate_assessment_node(state: AgentState) -> dict:
    print("Agent: Generating final assessment...")
    
    img_res = state.get("image_prediction") or {}
    sym_res = state.get("symptom_prediction") or {}
    docs = state.get("retrieved_documents") or []
    symptoms = state.get("symptoms") or []
    
    # Check if Google GenAI Key is available
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    
    if api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            context = f"""
            User Query: {state.get('user_query')}
            Identified Symptoms: {symptoms}
            Image Prediction Model Output: {img_res}
            Symptom Prediction Model Output: {sym_res}
            Retrieved Veterinary Knowledge: {docs}
            
            Please fuse the image model prediction and symptom model prediction. 
            Provide a grounded assessment, supporting evidence, and recommendations.
            Do NOT fabricate confidence values.
            """
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0, google_api_key=api_key)
            structured_llm = llm.with_structured_output(AssessmentOutput)
            response = structured_llm.invoke(context)
            assessment_dict = response.model_dump()
            return {
                "final_assessment": {
                    "assessment": {
                        "predicted_condition": assessment_dict["predicted_condition"],
                        "confidence": assessment_dict["confidence"],
                        "certainty_level": assessment_dict["certainty_level"]
                    },
                    "evidence": assessment_dict["evidence"],
                    "model_results": {
                        "image_model": img_res,
                        "symptom_model": sym_res
                    },
                    "knowledge_sources": assessment_dict["knowledge_sources"],
                    "recommendations": assessment_dict["recommendations"],
                    "warning": assessment_dict["warning"]
                }
            }
        except Exception as e:
            print(f"Agent LLM invocation error: {e}, falling back to intelligent rule-based fusion.")

    # Robust Intelligent Multimodal Fusion Fallback
    final_pred = "Undetermined Condition"
    confidence = "N/A"
    evidence = []
    
    if sym_res.get("prediction"):
        final_pred = sym_res["prediction"]
        confidence = "High (from symptom ensemble)"
        evidence.append(f"Symptom classifier ({sym_res.get('model', 'ML')}) indicated '{final_pred}' based on symptoms: {', '.join(symptoms)}.")
        
    if img_res.get("prediction"):
        img_p = img_res["prediction"]
        img_c = img_res.get("confidence", "Unknown")
        if not sym_res.get("prediction") or img_p.lower() == str(final_pred).lower():
            final_pred = img_p
            confidence = f"{img_c}% (Visual Analysis)"
            evidence.append(f"CNN model identified visual markers characteristic of '{img_p}' with {img_c}% confidence.")
        else:
            evidence.append(f"Visual CNN model suggested '{img_p}' ({img_c}%), but symptom model predicted '{final_pred}'.")
            
    if not evidence:
        evidence.append("No definitive symptoms or cattle images were detected in the input.")
        final_pred = "Inconclusive / More Information Needed"
        confidence = "Low"

    recs = [
        "Isolate the affected cattle if contagious symptoms (fever, lesions, severe diarrhoea) are observed.",
        "Ensure continuous access to clean water and appropriate nutrition.",
        "Consult a certified local veterinarian for clinical physical examination."
    ]

    return {
        "final_assessment": {
            "assessment": {
                "predicted_condition": final_pred,
                "confidence": str(confidence),
                "certainty_level": "preliminary"
            },
            "evidence": evidence,
            "model_results": {
                "image_model": img_res,
                "symptom_model": sym_res
            },
            "knowledge_sources": docs,
            "recommendations": recs,
            "warning": "This is an AI-assisted preliminary assessment and should not replace veterinary examination."
        }
    }

