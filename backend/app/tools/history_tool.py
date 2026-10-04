from langchain.tools import tool
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.database.models import Base, Case

# Setup SQLite database for local dev
engine = create_engine('sqlite:///cattle_health.db', connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@tool
def get_case_history(animal_id: str) -> dict:
    """
    Retrieves the medical history and previous cases for a given animal ID.
    Args:
        animal_id (str): The unique identifier of the animal.
    Returns:
        dict: A dictionary containing past symptoms, predictions, and case statuses.
    """
    db = SessionLocal()
    try:
        cases = db.query(Case).filter(Case.animal_id == animal_id).all()
        if not cases:
            return {"message": f"No case history found for animal {animal_id}."}
        
        history = []
        for case in cases:
            case_data = {
                "case_id": case.id,
                "created_at": str(case.created_at),
                "status": case.status,
                "symptoms": [s.symptom_list for s in case.symptoms] if case.symptoms else [],
                "predictions": [{"image": p.image_prediction, "symptom": p.symptom_prediction, "fusion": p.fusion_prediction} for p in case.predictions] if case.predictions else []
            }
            history.append(case_data)
        return {"history": history}
    except Exception as e:
        return {"error": str(e)}
    finally:
        db.close()
