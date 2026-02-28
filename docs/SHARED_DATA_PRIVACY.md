## Shared Data Privacy and Sharing Model

- **Per-user ownership**: all conversations, messages, safety events, and analysis artifacts are associated with a `User` row when synced to the backend.
- **Default visibility**:
  - Conversations are created with `visibility = "private"`.
  - `SafetyEvent` and `AnalysisArtifact` rows default to `share_scope = "aggregated_only"`, meaning they may be used in aggregate statistics but not exposed individually.
- **Sharing scopes**:
  - **private**: only the owning user can access the row.
  - **aggregated_only**: row may be included in anonymized aggregates (e.g., counts, averages) but not individually exposed.
  - **full_opt_in**: row may be used both in aggregates and, in future, for opt-in research or debugging views.
- **User preferences**:
  - The `User` model stores `share_safety_aggregated` and `share_conversations_anon` flags to control defaults and potential future behavior.
- **Aggregated analytics**:
  - The `/api/v1/conversations/safety/summary` endpoint only includes safety events where `share_scope != "private"`.

