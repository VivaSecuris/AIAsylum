"""Add weight-surgery runs

Revision ID: 007
Revises: 006
Create Date: 2026-09-23 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '007'
down_revision = '006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    from sqlalchemy import inspect

    conn = op.get_bind()
    inspector = inspect(conn)

    if 'weight_runs' not in inspector.get_table_names():
        op.create_table(
            'weight_runs',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), nullable=True),
            # direction | sweep | surgery
            sa.Column('kind', sa.String(length=16), nullable=False),
            sa.Column('status', sa.String(length=20), nullable=True),
            sa.Column('source_model', sa.String(length=512), nullable=False),
            # The direction run a sweep or surgery consumed. Deliberately not a
            # ForeignKey: deleting a direction that has children is allowed, and
            # the children snapshot what they need at creation time.
            sa.Column('source_run_id', sa.Integer(), nullable=True),
            sa.Column('method', sa.String(length=32), nullable=True),
            sa.Column('objective', sa.String(length=32), nullable=True),
            sa.Column('out_dir', sa.String(length=1024), nullable=True),
            sa.Column('artifact_bytes', sa.BigInteger(), nullable=True),
            sa.Column('started_at', sa.DateTime(), nullable=True),
            sa.Column('completed_at', sa.DateTime(), nullable=True),
            sa.Column('error', sa.Text(), nullable=True),
            # Column is 'metadata' in SQL, mapped to meta_data in Python to avoid
            # colliding with SQLAlchemy's Declarative attribute of that name.
            sa.Column('metadata', sa.JSON(), nullable=True),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index(op.f('ix_weight_runs_id'), 'weight_runs', ['id'], unique=False)
        op.create_index(op.f('ix_weight_runs_kind'), 'weight_runs', ['kind'], unique=False)
        op.create_index(op.f('ix_weight_runs_status'), 'weight_runs', ['status'], unique=False)
        op.create_index(
            op.f('ix_weight_runs_source_run_id'), 'weight_runs', ['source_run_id'], unique=False
        )


def downgrade() -> None:
    from sqlalchemy import inspect

    conn = op.get_bind()
    inspector = inspect(conn)

    if 'weight_runs' in inspector.get_table_names():
        op.drop_index(op.f('ix_weight_runs_source_run_id'), table_name='weight_runs')
        op.drop_index(op.f('ix_weight_runs_status'), table_name='weight_runs')
        op.drop_index(op.f('ix_weight_runs_kind'), table_name='weight_runs')
        op.drop_index(op.f('ix_weight_runs_id'), table_name='weight_runs')
        op.drop_table('weight_runs')
