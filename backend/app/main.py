from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import shutil
import os
import uuid

from backend.app.agent.graph import agent_executor

app = FastAPI(title="Multimodal Agentic AI Cattle Health API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/agent/diagnose")
async def diagnose(
    user_query: Optional[str] = Form(""),
    symptoms: Optional[List[str]] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    image_path = None
    if image:
        filename = f"{uuid.uuid4()}_{image.filename}"
        image_path = os.path.join(UPLOAD_DIR, filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
            
    # Default symptoms if None
    if not symptoms:
        symptoms = []
        
    # Execute LangGraph Workflow
    inputs = {
        "user_query": user_query,
        "symptoms": symptoms,
        "image_path": image_path,
    }
    
    # Run the graph
    print("Agent: Starting workflow execution...")
    result = agent_executor.invoke(inputs)
    
    return result.get("final_assessment", {"error": "Assessment failed"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
