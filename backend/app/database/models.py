from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

class Animal(Base):
    __tablename__ = 'animals'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    age = Column(Integer, nullable=True)
    breed = Column(String, nullable=True)
    
    cases = relationship("Case", back_populates="animal")

class Case(Base):
    __tablename__ = 'cases'
    
    id = Column(String, primary_key=True)
    animal_id = Column(String, ForeignKey('animals.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="open")
    
    animal = relationship("Animal", back_populates="cases")
    symptoms = relationship("Symptom", back_populates="case")
    predictions = relationship("Prediction", back_populates="case")
    sessions = relationship("AgentSession", back_populates="case")

class Symptom(Base):
    __tablename__ = 'symptoms'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String, ForeignKey('cases.id'))
    symptom_list = Column(JSON) # List of symptoms
    
    case = relationship("Case", back_populates="symptoms")

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String, ForeignKey('cases.id'))
    image_prediction = Column(JSON, nullable=True)
    symptom_prediction = Column(JSON, nullable=True)
    fusion_prediction = Column(JSON, nullable=True)
    
    case = relationship("Case", back_populates="predictions")

class AgentSession(Base):
    __tablename__ = 'agent_sessions'
    
    id = Column(String, primary_key=True)
    case_id = Column(String, ForeignKey('cases.id'))
    messages = Column(JSON)
    tool_calls = Column(JSON)
    
    case = relationship("Case", back_populates="sessions")
