"""Add interpretability runs

Revision ID: 006
Revises: 005
Create Date: 2026-09-23 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    from sqlalchemy import inspect

    conn = op.get_bind()
    inspector = inspect(conn)

    if 'interp_runs' not in inspector.get_table_names():
        op.create_table(
            'interp_runs',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), nullable=True),
            sa.Column('mode', sa.String(length=32), nullable=False),
            sa.Column('status', sa.String(length=20), nullable=True),
            sa.Column('model_a', sa.String(length=512), nullable=False),
            sa.Column('model_b', sa.String(length=512), nullable=True),
            sa.Column('provider', sa.String(length=50), nullable=True),
            sa.Column('prompt_a', sa.Text(), nullable=True),
            sa.Column('prompt_b', sa.Text(), nullable=True),
            sa.Column('prompts', sa.JSON(), nullable=True),
            sa.Column('out_dir', sa.String(length=1024), nullable=True),
            sa.Column('started_at', sa.DateTime(), nullable=True),
            sa.Column('completed_at', sa.DateTime(), nullable=True),
            sa.Column('error', sa.Text(), nullable=True),
            # Column is 'metadata' in SQL, mapped to meta_data in Python to avoid
            # colliding with SQLAlchemy's Declarative attribute of that name.
            sa.Column('metadata', sa.JSON(), nullable=True),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index(op.f('ix_interp_runs_id'), 'interp_runs', ['id'], unique=False)
        op.create_index(op.f('ix_interp_runs_mode'), 'interp_runs', ['mode'], unique=False)
        op.create_index(op.f('ix_interp_runs_status'), 'interp_runs', ['status'], unique=False)


def downgrade() -> None:
    from sqlalchemy import inspect

    conn = op.get_bind()
    inspector = inspect(conn)

    if 'interp_runs' in inspector.get_table_names():
        op.drop_index(op.f('ix_interp_runs_status'), table_name='interp_runs')
        op.drop_index(op.f('ix_interp_runs_mode'), table_name='interp_runs')
        op.drop_index(op.f('ix_interp_runs_id'), table_name='interp_runs')
        op.drop_table('interp_runs')
