from langgraph.graph import StateGraph, END
from backend.app.agent.state import AgentState
from backend.app.agent.nodes import (
    analyze_image_node,
    analyze_symptoms_node,
    retrieve_knowledge_node,
    generate_assessment_node
)

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("analyze_image", analyze_image_node)
    workflow.add_node("analyze_symptoms", analyze_symptoms_node)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge_node)
    workflow.add_node("generate_assessment", generate_assessment_node)
    
    # Workflow routing logic:
    # We can run image and symptom analysis in parallel (or sequential)
    # Then retrieve knowledge, then generate assessment.
    
    workflow.set_entry_point("analyze_image")
    workflow.add_edge("analyze_image", "analyze_symptoms")
    workflow.add_edge("analyze_symptoms", "retrieve_knowledge")
    workflow.add_edge("retrieve_knowledge", "generate_assessment")
    workflow.add_edge("generate_assessment", END)
    
    return workflow.compile()

agent_executor = build_graph()
