"""Database models."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class TestRun(Base):
    """A test run session."""
    
    __tablename__ = "test_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Model information
    doctor_provider = Column(String(50), nullable=False)
    doctor_model = Column(String(100), nullable=False)
    patient_provider = Column(String(50), nullable=False)
    patient_model = Column(String(100), nullable=False)
    
    # Test configuration
    test_type = Column(String(50), nullable=False)  # conversation, scenario, adversarial
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    
    # Results
    results = relationship("TestResult", back_populates="test_run", cascade="all, delete-orphan")
    conversations = relationship("ConversationTurn", back_populates="test_run", cascade="all, delete-orphan")
    assessments = relationship("Assessment", back_populates="test_run", cascade="all, delete-orphan")
    
    # Metadata (renamed to avoid SQLAlchemy conflict)
    meta_data = Column("metadata", JSON, default=dict)
    
    def __repr__(self):
        return f"<TestRun(id={self.id}, test_type={self.test_type}, status={self.status})>"


class TestResult(Base):
    """Individual test result."""
    
    __tablename__ = "test_results"
    
    id = Column(Integer, primary_key=True, index=True)
    test_run_id = Column(Integer, ForeignKey("test_runs.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Test details
    test_name = Column(String(200), nullable=False)
    test_category = Column(String(50))  # safety, reasoning, alignment, etc.
    
    # Input/Output
    input_prompt = Column(Text, nullable=False)
    output_response = Column(Text, nullable=False)
    
    # Scoring
    score = Column(Float)  # 0.0 to 1.0
    scores = Column(JSON)  # Detailed scores by dimension
    
    # Analysis
    analysis = Column(Text)
    flags = Column(JSON)  # List of flags (e.g., ["harmful", "jailbreak_attempt"])
    
    # Metadata (renamed to avoid SQLAlchemy conflict)
    meta_data = Column("metadata", JSON, default=dict)
    
    test_run = relationship("TestRun", back_populates="results")
    
    def __repr__(self):
        return f"<TestResult(id={self.id}, test_name={self.test_name}, score={self.score})>"


class ConversationTurn(Base):
    """A single turn in a conversation test."""
    
    __tablename__ = "conversation_turns"
    
    id = Column(Integer, primary_key=True, index=True)
    test_run_id = Column(Integer, ForeignKey("test_runs.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Turn information
    turn_number = Column(Integer, nullable=False)
    speaker = Column(String(20), nullable=False)  # doctor or patient
    
    # Content
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    
    # Model information
    model_provider = Column(String(50))
    model_name = Column(String(100))
    
    # Usage stats
    usage = Column(JSON)  # Token usage, etc.
    
    # Metadata (renamed to avoid SQLAlchemy conflict)
    meta_data = Column("metadata", JSON, default=dict)
    
    test_run = relationship("TestRun", back_populates="conversations")
    
    def __repr__(self):
        return f"<ConversationTurn(id={self.id}, turn={self.turn_number}, speaker={self.speaker})>"


class Assessment(Base):
    """Doctor's assessment of a test run."""
    
    __tablename__ = "assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    test_run_id = Column(Integer, ForeignKey("test_runs.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Assessment content
    assessment_text = Column(Text, nullable=False)
    
    # Scores by dimension
    scores = Column(JSON, nullable=False)  # {"alignment": 0.8, "safety": 0.7, ...}
    overall_score = Column(Float, nullable=False)  # Weighted average
    
    # Analysis type
    analysis_type = Column(String(50))  # initial, final, deep_analysis
    
    # Flags and concerns
    flags = Column(JSON, default=list)
    concerns = Column(Text)
    recommendations = Column(Text)
    
    # Metadata (renamed to avoid SQLAlchemy conflict)
    meta_data = Column("metadata", JSON, default=dict)
    
    test_run = relationship("TestRun", back_populates="assessments")
    
    def __repr__(self):
        return f"<Assessment(id={self.id}, overall_score={self.overall_score})>"


class BenchmarkResult(Base):
    """Results from standard benchmark evaluations."""
    
    __tablename__ = "benchmark_results"
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Model information
    model_provider = Column(String(50), nullable=False)
    model_name = Column(String(100), nullable=False)
    
    # Benchmark information
    benchmark_name = Column(String(50), nullable=False)  # mmlu, truthfulqa, etc.
    benchmark_version = Column(String(20))
    
    # Results
    score = Column(Float, nullable=False)  # Overall score
    scores_by_category = Column(JSON)  # Scores by category/subset
    num_samples = Column(Integer, nullable=False)
    num_correct = Column(Integer)
    
    # Details
    results = Column(JSON)  # Detailed results per sample
    metadata = Column(JSON, default=dict)
    
    def __repr__(self):
        return f"<BenchmarkResult(id={self.id}, benchmark={self.benchmark_name}, score={self.score})>"
