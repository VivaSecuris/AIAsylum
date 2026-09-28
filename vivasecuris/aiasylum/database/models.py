"""Database models."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
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
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import relationship

# JSON metadata that tracks top-level key changes: without this, code that writes
# ``row.meta_data["name"] = ...`` is silently never saved. Nested edits still need
# the dict (or the nested value) reassigned.
MetaJSON = MutableDict.as_mutable(JSON)

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
    
    # Suite relationship
    suite_id = Column(Integer, ForeignKey("test_suites.id"), nullable=True, index=True)
    suite = relationship("TestSuite", back_populates="test_runs")
    
    # Results
    results = relationship("TestResult", back_populates="test_run", cascade="all, delete-orphan")
    conversations = relationship("ConversationTurn", back_populates="test_run", cascade="all, delete-orphan")
    assessments = relationship("Assessment", back_populates="test_run", cascade="all, delete-orphan")
    
    # Metadata (renamed to avoid SQLAlchemy conflict)
    meta_data = Column("metadata", MetaJSON, default=dict)
    
    def __repr__(self) -> str:
        return f"<TestRun(id={self.id}, test_type={self.test_type}, status={self.status})>"
    
    def __str__(self) -> str:
        return f"TestRun #{self.id}: {self.test_type} ({self.status})"


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
    meta_data = Column("metadata", MetaJSON, default=dict)
    
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
    meta_data = Column("metadata", MetaJSON, default=dict)
    
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
    meta_data = Column("metadata", MetaJSON, default=dict)
    
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
    # Metadata (renamed to avoid SQLAlchemy conflict)
    meta_data = Column("metadata", MetaJSON, default=dict)
    
    def __repr__(self):
        return f"<BenchmarkResult(id={self.id}, benchmark={self.benchmark_name}, score={self.score})>"


class PromptLibrary(Base):
    """User prompts library for testing multiple models."""
    
    __tablename__ = "prompt_library"
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Prompt information
    name = Column(String(200), nullable=False)
    description = Column(Text)
    prompt_text = Column(Text, nullable=False)
    
    # Prompt type and target
    prompt_type = Column(String(50), default="test_prompt")  # test_prompt or system_prompt
    target = Column(String(50))  # doctor, patient, or None for test prompts
    
    # Categorization
    category = Column(String(50))  # conversation, adversarial, scenario, etc.
    tags = Column(JSON, default=list)  # List of tags for filtering
    
    # Usage tracking
    usage_count = Column(Integer, default=0)  # Number of times used in test runs
    
    # Metadata (renamed to avoid SQLAlchemy conflict)
    meta_data = Column("metadata", MetaJSON, default=dict)
    
    def __repr__(self):
        return f"<PromptLibrary(id={self.id}, name={self.name}, type={self.prompt_type}, target={self.target})>"


class TestSuite(Base):
    """A test suite containing multiple test runs."""
    
    __tablename__ = "test_suites"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200))  # Optional suite name
    status = Column(String(20), default="pending")  # pending, running, completed, failed, partially_failed
    
    # Progress tracking
    total_runs = Column(Integer, default=0)
    completed_runs = Column(Integer, default=0)
    failed_runs = Column(Integer, default=0)
    running_runs = Column(Integer, default=0)
    pending_runs = Column(Integer, default=0)
    
    # Time tracking
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Metadata (renamed to avoid SQLAlchemy conflict)
    meta_data = Column("metadata", MetaJSON, default=dict)
    
    # Relationship to test runs
    test_runs = relationship("TestRun", back_populates="suite", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<TestSuite(id={self.id}, name={self.name}, status={self.status}, total_runs={self.total_runs})>"
    
    def __str__(self) -> str:
        return f"TestSuite #{self.id}: {self.name or 'Unnamed'} ({self.status})"


class User(Base):
    """Represents an end user or installation using the system.

    Note: current authentication is API-key based via settings.
    This table is primarily for ownership, preferences, and future multi-tenant support.
    """

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Optional association with an API key used for authentication
    api_key = Column(String(128), unique=True, nullable=True, index=True)

    # Optional display information
    display_name = Column(String(100), nullable=True)

    # Per-user preferences
    ollama_base_url = Column(String(255), nullable=True)

    # Privacy / sharing preferences
    share_safety_aggregated = Column(Boolean, default=False, nullable=False)
    share_conversations_anon = Column(Boolean, default=False, nullable=False)


class Conversation(Base):
    """Logical conversation between a user and one or more models."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Ownership
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)

    title = Column(String(255), nullable=True)

    # Visibility controls: private, org, public_anon (for future org support)
    visibility = Column(String(32), default="private", nullable=False)

    # Free-form metadata about the conversation
    meta_data = Column("metadata", MetaJSON, default=dict)

    # Relationships with cascade so deleting a conversation removes all related data
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=False,
    )
    safety_events = relationship(
        "SafetyEvent",
        back_populates="conversation",
        foreign_keys="SafetyEvent.conversation_id",
        cascade="all, delete-orphan",
    )
    analysis_artifacts = relationship(
        "AnalysisArtifact",
        back_populates="conversation",
        foreign_keys="AnalysisArtifact.conversation_id",
        cascade="all, delete-orphan",
    )


class Message(Base):
    """Individual message within a conversation."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)

    # user / assistant / system / tool etc.
    role = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)

    # Model and run metadata when this message comes from a model
    model_provider = Column(String(50), nullable=True)
    model_name = Column(String(100), nullable=True)
    usage = Column(JSON, nullable=True)

    meta_data = Column("metadata", MetaJSON, default=dict)

    conversation = relationship("Conversation", back_populates="messages")
    safety_events = relationship(
        "SafetyEvent",
        back_populates="message",
        foreign_keys="SafetyEvent.message_id",
        cascade="all, delete-orphan",
    )
    analysis_artifacts = relationship(
        "AnalysisArtifact",
        back_populates="message",
        foreign_keys="AnalysisArtifact.message_id",
        cascade="all, delete-orphan",
    )


class SafetyEvent(Base):
    """Safety-related signal associated with a message or conversation."""

    __tablename__ = "safety_events"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Scope
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)

    # What was evaluated
    model_provider = Column(String(50), nullable=True)
    model_name = Column(String(100), nullable=True)
    check_type = Column(String(64), nullable=False)  # e.g. toxicity, jailbreak, pii

    # Result
    score = Column(Float, nullable=True)
    label = Column(String(32), nullable=True)  # safe / unsafe / needs_review etc.
    raw_model_output = Column(Text, nullable=True)

    # Sharing scope for this event: private, aggregated_only, full_opt_in
    share_scope = Column(String(32), default="aggregated_only", nullable=False, index=True)

    meta_data = Column("metadata", MetaJSON, default=dict)

    conversation = relationship(
        "Conversation",
        back_populates="safety_events",
        foreign_keys=[conversation_id],
    )
    message = relationship(
        "Message",
        back_populates="safety_events",
        foreign_keys=[message_id],
    )


class AnalysisArtifact(Base):
    """Arbitrary analysis artifact attached to a message or conversation."""

    __tablename__ = "analysis_artifacts"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Scope
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)

    # Artifact details
    type = Column(String(64), nullable=False)  # e.g. chain_of_thought_summary, classifier_features
    payload = Column(JSON, nullable=False)  # Arbitrary JSON content

    # Sharing scope: private, aggregated_only, full_opt_in
    share_scope = Column(String(32), default="aggregated_only", nullable=False, index=True)

    meta_data = Column("metadata", MetaJSON, default=dict)

    conversation = relationship(
        "Conversation",
        back_populates="analysis_artifacts",
        foreign_keys=[conversation_id],
    )
    message = relationship(
        "Message",
        back_populates="analysis_artifacts",
        foreign_keys=[message_id],
    )


class InterpRun(Base):
    """A mechanistic-interpretability analysis of one or two models.

    Kept separate from TestRun rather than reusing it with a test_type: an
    interp run has no doctor and no patient, so four of TestRun's NOT NULL
    columns would carry placeholder values. A separate table also gives interp
    runs their own id space, which matters because the progress and
    cancellation managers are keyed by bare integers.
    """

    __tablename__ = "interp_runs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # single | comparison | progression | model_diff
    mode = Column(String(32), nullable=False, index=True)
    status = Column(String(20), default="pending", index=True)

    # model_b is set only for model_diff (e.g. baseline vs ablated).
    model_a = Column(String(512), nullable=False)
    model_b = Column(String(512), nullable=True)
    provider = Column(String(50), default="transformers")

    # prompt_a/prompt_b for single and comparison; prompts for progression.
    prompt_a = Column(Text, nullable=True)
    prompt_b = Column(Text, nullable=True)
    prompts = Column(JSON, nullable=True)

    # Where the artifacts and dashboard.html were written.
    out_dir = Column(String(1024), nullable=True)

    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error = Column(Text, nullable=True)

    meta_data = Column("metadata", MetaJSON, default=dict)

    def __repr__(self) -> str:
        return f"<InterpRun(id={self.id}, mode={self.mode}, status={self.status})>"

    def __str__(self) -> str:
        return f"InterpRun #{self.id}: {self.mode} ({self.status})"


class WeightRun(Base):
    """One stage of the weight-surgery pipeline: derive, sweep, or surgery.

    A single table with a ``kind`` discriminator rather than three. The stages
    share status, timestamps, an output directory, an SSE stream and a
    cancellation id space, and they chain -- so they want one id space. Three
    tables would force a UNION for the run feed and a polymorphic id for the
    progress manager, which keys on a bare integer.

    ``kind`` is ``surgery`` rather than ``ablate`` because beta selects the
    behaviour: 0 ablates, 1 is a bit-identical control, 2 amplifies. Naming the
    stage after one of its three settings reads as a bug the first time someone
    runs an amplification control.

    ``source_run_id`` points at the direction a sweep or surgery consumed, and
    is deliberately **not** a ForeignKey: interp_runs set that precedent,
    SQLite does not enforce FKs by default, and RESTRICT would block deleting a
    direction that has children -- which is allowed here. The child instead
    snapshots what it needs into ``meta_data["source_direction"]``, so its page
    still renders once the parent is gone. That snapshot is also forced by
    RefusalDirection.load(), which silently drops layer_scores: a round trip
    through the loader loses the whole AUC curve.
    """

    __tablename__ = "weight_runs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # direction | sweep | surgery
    kind = Column(String(16), nullable=False, index=True)
    status = Column(String(20), default="pending", index=True)

    # The model being read: a Hugging Face id or a local directory.
    source_model = Column(String(512), nullable=False)
    source_run_id = Column(Integer, nullable=True, index=True)

    # Which edit, and what behaviour it targets. Recorded rather than assumed:
    # the pipeline defaults to direction_scale on refusal, but neither is a
    # constraint of the engine, and a manifest that omits them cannot say what
    # a modified model was actually aimed at.
    method = Column(String(32), default="direction_scale")
    objective = Column(String(32), default="refusal")

    # direction/sweep: runs/weights/<id>. surgery: models/<name>.
    out_dir = Column(String(1024), nullable=True)
    artifact_bytes = Column(BigInteger, nullable=True)

    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error = Column(Text, nullable=True)

    meta_data = Column("metadata", MetaJSON, default=dict)

    def __repr__(self) -> str:
        return f"<WeightRun(id={self.id}, kind={self.kind}, status={self.status})>"

    def __str__(self) -> str:
        return f"WeightRun #{self.id}: {self.kind} ({self.status})"
