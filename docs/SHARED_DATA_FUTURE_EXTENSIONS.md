## Future Extensions: Orgs/Teams and Vector Search

### Organizations and teams

- Introduce `organizations` and `user_org_memberships` tables:
  - `organizations`: `id`, `name`, `created_at`, `updated_at`, optional metadata.
  - `user_org_memberships`: `user_id`, `org_id`, `role`, `created_at`.
- Extend ownership and visibility:
  - Add optional `org_id` columns to `Conversation`, `SafetyEvent`, and `AnalysisArtifact`.
  - Extend `visibility` to `private`, `org`, `public_anon`.
- Access control:
  - Enforce org-based access in application code (or later via row-level security when running on PostgreSQL).

### Vector search and shared intelligence

- Add an `embeddings` table when running on PostgreSQL with pgvector:
  - `id`, `owner_user_id`, optional `org_id`.
  - `source_type` (message, safety_event, analysis_artifact) and `source_id`.
  - `vector` column using `pgvector`.
  - `metadata` JSON with visibility and redaction status.
- Generation strategy:
  - Compute embeddings either:
    - On-device and sync via the existing `/api/v1/conversations/sync` pipeline, or
    - In a background worker on the backend for data explicitly opted in by the user.
- Query API:
  - Expose a search endpoint that:
    - Accepts a query embedding or text to embed.
    - Filters candidates by `owner_user_id`, `org_id`, and visibility/share_scope.
    - Returns only content whose visibility permits it (e.g. private for the owner, org-visible for teammates, aggregated or anonymized for global stats).

