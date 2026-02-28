"""Add shared conversations and safety/analysis tables

Revision ID: 005
Revises: 004
Create Date: 2026-02-27 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Users table
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("api_key", sa.String(length=128), nullable=True),
        sa.Column("display_name", sa.String(length=100), nullable=True),
        sa.Column("ollama_base_url", sa.String(length=255), nullable=True),
        sa.Column("share_safety_aggregated", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("share_conversations_anon", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_id"), "users", ["id"], unique=False)
    op.create_index(op.f("ix_users_api_key"), "users", ["api_key"], unique=True)

    # Conversations table
    op.create_table(
        "conversations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("visibility", sa.String(length=32), nullable=False, server_default="private"),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_conversations_id"), "conversations", ["id"], unique=False)
    op.create_index(op.f("ix_conversations_user_id"), "conversations", ["user_id"], unique=False)

    # Messages table
    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("model_provider", sa.String(length=50), nullable=True),
        sa.Column("model_name", sa.String(length=100), nullable=True),
        sa.Column("usage", sa.JSON(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_messages_id"), "messages", ["id"], unique=False)
    op.create_index(op.f("ix_messages_conversation_id"), "messages", ["conversation_id"], unique=False)
    op.create_index(op.f("ix_messages_user_id"), "messages", ["user_id"], unique=False)

    # Safety events
    op.create_table(
        "safety_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("message_id", sa.Integer(), nullable=True),
        sa.Column("conversation_id", sa.Integer(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("model_provider", sa.String(length=50), nullable=True),
        sa.Column("model_name", sa.String(length=100), nullable=True),
        sa.Column("check_type", sa.String(length=64), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("label", sa.String(length=32), nullable=True),
        sa.Column("raw_model_output", sa.Text(), nullable=True),
        sa.Column(
            "share_scope",
            sa.String(length=32),
            nullable=False,
            server_default="aggregated_only",
        ),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_safety_events_id"), "safety_events", ["id"], unique=False)
    op.create_index(op.f("ix_safety_events_message_id"), "safety_events", ["message_id"], unique=False)
    op.create_index(op.f("ix_safety_events_conversation_id"), "safety_events", ["conversation_id"], unique=False)
    op.create_index(op.f("ix_safety_events_user_id"), "safety_events", ["user_id"], unique=False)
    op.create_index(op.f("ix_safety_events_share_scope"), "safety_events", ["share_scope"], unique=False)

    # Analysis artifacts
    op.create_table(
        "analysis_artifacts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("message_id", sa.Integer(), nullable=True),
        sa.Column("conversation_id", sa.Integer(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("type", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column(
            "share_scope",
            sa.String(length=32),
            nullable=False,
            server_default="aggregated_only",
        ),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_analysis_artifacts_id"), "analysis_artifacts", ["id"], unique=False)
    op.create_index(
        op.f("ix_analysis_artifacts_message_id"),
        "analysis_artifacts",
        ["message_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_analysis_artifacts_conversation_id"),
        "analysis_artifacts",
        ["conversation_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_analysis_artifacts_user_id"),
        "analysis_artifacts",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_analysis_artifacts_share_scope"),
        "analysis_artifacts",
        ["share_scope"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_analysis_artifacts_share_scope"), table_name="analysis_artifacts")
    op.drop_index(op.f("ix_analysis_artifacts_user_id"), table_name="analysis_artifacts")
    op.drop_index(op.f("ix_analysis_artifacts_conversation_id"), table_name="analysis_artifacts")
    op.drop_index(op.f("ix_analysis_artifacts_message_id"), table_name="analysis_artifacts")
    op.drop_index(op.f("ix_analysis_artifacts_id"), table_name="analysis_artifacts")
    op.drop_table("analysis_artifacts")

    op.drop_index(op.f("ix_safety_events_share_scope"), table_name="safety_events")
    op.drop_index(op.f("ix_safety_events_user_id"), table_name="safety_events")
    op.drop_index(op.f("ix_safety_events_conversation_id"), table_name="safety_events")
    op.drop_index(op.f("ix_safety_events_message_id"), table_name="safety_events")
    op.drop_index(op.f("ix_safety_events_id"), table_name="safety_events")
    op.drop_table("safety_events")

    op.drop_index(op.f("ix_messages_user_id"), table_name="messages")
    op.drop_index(op.f("ix_messages_conversation_id"), table_name="messages")
    op.drop_index(op.f("ix_messages_id"), table_name="messages")
    op.drop_table("messages")

    op.drop_index(op.f("ix_conversations_user_id"), table_name="conversations")
    op.drop_index(op.f("ix_conversations_id"), table_name="conversations")
    op.drop_table("conversations")

    op.drop_index(op.f("ix_users_api_key"), table_name="users")
    op.drop_index(op.f("ix_users_id"), table_name="users")
    op.drop_table("users")

