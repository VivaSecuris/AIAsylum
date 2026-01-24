"""Add system prompt support

Revision ID: 003
Revises: 002
Create Date: 2025-01-24 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add prompt_type and target columns to prompt_library
    op.add_column('prompt_library', sa.Column('prompt_type', sa.String(length=50), nullable=True))
    op.add_column('prompt_library', sa.Column('target', sa.String(length=50), nullable=True))
    
    # Set default value for existing rows
    op.execute("UPDATE prompt_library SET prompt_type = 'test_prompt' WHERE prompt_type IS NULL")


def downgrade() -> None:
    op.drop_column('prompt_library', 'target')
    op.drop_column('prompt_library', 'prompt_type')
