"""Add test suites

Revision ID: 004
Revises: 003
Create Date: 2025-01-25 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create test_suites table (if it doesn't exist)
    from sqlalchemy import inspect
    conn = op.get_bind()
    inspector = inspect(conn)
    tables = inspector.get_table_names()
    
    if 'test_suites' not in tables:
        op.create_table(
            'test_suites',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('name', sa.String(length=200), nullable=True),
            sa.Column('status', sa.String(length=20), nullable=True),
            sa.Column('total_runs', sa.Integer(), nullable=True),
            sa.Column('completed_runs', sa.Integer(), nullable=True),
            sa.Column('failed_runs', sa.Integer(), nullable=True),
            sa.Column('running_runs', sa.Integer(), nullable=True),
            sa.Column('pending_runs', sa.Integer(), nullable=True),
            sa.Column('started_at', sa.DateTime(), nullable=True),
            sa.Column('completed_at', sa.DateTime(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), nullable=True),
            sa.Column('metadata', sa.JSON(), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_test_suites_id'), 'test_suites', ['id'], unique=False)
    
    # Check if suite_id column already exists
    test_runs_columns = [col['name'] for col in inspector.get_columns('test_runs')]
    
    # Add suite_id column to test_runs (SQLite requires batch mode for ALTER TABLE)
    if 'suite_id' not in test_runs_columns:
        with op.batch_alter_table('test_runs', schema=None) as batch_op:
            batch_op.add_column(sa.Column('suite_id', sa.Integer(), nullable=True))
            batch_op.create_index(op.f('ix_test_runs_suite_id'), ['suite_id'], unique=False)
            # Note: SQLite doesn't support adding foreign keys via ALTER TABLE
            # The foreign key constraint will be enforced at the application level


def downgrade() -> None:
    # SQLite requires batch mode for ALTER TABLE
    with op.batch_alter_table('test_runs', schema=None) as batch_op:
        batch_op.drop_constraint('fk_test_runs_suite_id', type_='foreignkey')
        batch_op.drop_index(op.f('ix_test_runs_suite_id'))
        batch_op.drop_column('suite_id')
    op.drop_index(op.f('ix_test_suites_id'), table_name='test_suites')
    op.drop_table('test_suites')
