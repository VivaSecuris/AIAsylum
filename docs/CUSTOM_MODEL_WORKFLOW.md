# Custom model workflow

## Work from the model library

Open **Models → Models** (`/compare?view=models`). Custom checkpoints and original models share a searchable library. A checkpoint appears before it has any test results; availability reflects files on the connected API server. Imported history can be visible while its model files are still missing. Missing or incomplete checkpoints cannot be selected for execution.

- **Compare with original** opens an activation comparison with both checkpoints selected. Add a shared prompt, check model fit, then launch.
- **Compare behavior** prepares response and capability evaluation using the same original/custom pair.
- **Analyze** selects one checkpoint in Interpretability. **Evaluate** selects it in the test form.
- **Charts & visualizations**, the default Models view, contains evaluation comparisons, saved interpretability dashboards, and weight experiment plots. The older `/model-results` link still opens evaluation charts directly.
- Saving a new weight edit makes it available in this library. Direction derivation and steering sweeps produce experiment artifacts; they do not themselves save a model checkpoint.

## Find saved graphs and visualizations

**Models → Charts & visualizations** restores the original evaluation chart families, including score comparison, ranking, heatmap, radar charts, test coverage/status distributions, factuality, manipulation resistance, and model statistics. Missing assessment scores are shown as unavailable instead of zero; request failures are visible and retryable.

Choose **Interpretability dashboards** to select a completed analysis and view its saved interactive charts directly. Its model pair, timestamp, and device identify the run. **Full analysis & causal tests** opens the run's detailed patching and circuit views when those artifacts were recorded. **Weight experiment plots** exposes recorded layer separation, steering, capability frontier, and comparison data, with a link to the full run.

The connected server is shown at the top. AWS and the local workstation retain separate run registries. Importing lineage does not import scored assessments or dashboard artifacts. On September 24, 2026, AWS had 22 completed interpretability dashboards and no scored behavioral assessments; the local workstation retained its earlier evaluation scores, two interpretability dashboards, and 23 completed weight runs. Use the local WebUI for those original results. The local and GPU frontends use separate build directories (`.next-local` and `.next-gpu`); production verification must use another `NEXT_DIST_DIR` to avoid corrupting a running development server's chunks.

## Find models and organize experiments

**Models → Find models** and **Find common models** in each server model picker list common Qwen, Llama, Gemma, Mistral, Phi, and SmolLM checkpoints. Search queries go to the public Hugging Face Hub; searching the local library stays local. Selecting a result only fills the model reference. **Download to server** explicitly starts a download, and the model appears as ready once its checkpoint files are present. Discovery is not proof of runtime compatibility or sufficient GPU memory; review the analysis preflight before running a new architecture or size.

Public repositories need no Hugging Face login. Gated and private models require an account with repository access and a read token configured for the account running the API on the analysis server. Browser sign-in alone does not connect AWS. The UI shows whether a token is configured (it does not validate that token or prove access). For gated models, open the linked model page to review its license and obtain access, then run `hf auth login` in the server's Python environment. Keep credentials out of model IDs, notes, and chat. Restart the API only when no jobs are active and refresh the access status.

Use **New experiment** to name a study and describe its purpose. **Organize · name, notes and experiments** on a model card or selected history step saves an optional display name, notes, and experiment memberships. A model or step can belong to more than one experiment. Select an experiment above the library or history to focus the workspace; **Unassigned models and steps** helps collect older work. Model pickers also support experiment filtering.

The history **Steps** view offers chronological browsing with name/notes search and a step-type filter. The dependency graph keeps the same recorded edges. When an experiment is selected, required ancestors remain visible as context even if they belong to another experiment. Display names do not rename files or rewrite run IDs, timestamps, parent relationships, or original recorded labels.

Organization is persisted on the connected server in `runs/model-organization.json`, separately from immutable history. Include this file in server backups. Its model references and step IDs are specific to that server; the lineage exporter does not currently remap or import organization metadata. New runs can be assigned through their history inspector after creation; experiment membership is not inferred automatically from a parent.

Live discovery and organization validation on September 24, 2026 downloaded `HuggingFaceTB/SmolLM2-135M-Instruct` anonymously to AWS (260 MiB, pinned snapshot `12fd25f77366fa6b3b4b768ec3050bf629380bac`). WebUI evaluation #1 and CUDA BF16 analysis #23 completed. Analysis captured 31 hidden states across five input tokens and released all PyTorch allocated/reserved GPU memory after completion. The model and both runs are grouped in **Public model workflow validation**, with saved display names and notes. These are workflow checks, not accuracy benchmarks or validation of every model in the discovery list.

## Inspect and branch from history

Open **Models → History graph**, or a checkpoint's **View history** action. The graph contains original checkpoints, derived directions, sweeps, searches, edits, saved checkpoints, activation analyses, and evaluations. Failed attempts remain visible.

Connections represent recorded inputs and outputs. Creation timestamps show order, but neighboring timestamps do not imply a dependency: a sweep and an edit using the same direction are siblings unless an explicit branch relationship was recorded. Missing parents are shown as missing steps instead of guessed connections. All timestamps are normalized from stored UTC before display in the browser's timezone.

Select a step to inspect its settings, results, provenance, inputs, and next steps. **Branch from this step** or **Review and retry as a new run** opens an editable form; it does not execute immediately. Starting that form creates a new run linked back to the selected step, preserving the original result. Saving an edited checkpoint requires a new output name.

Imported histories retain their origin namespace. An imported run numbered 12 never becomes AWS run 12. For an imported sweep/edit, the branch form first recreates its recorded direction on the current server; the inspector explains that transition. A checkpoint can be the source of a new direction directly once its files are ready. Missing historical inputs cannot be recovered by the graph alone.

New runs have durable lineage identities independent of their database integer. Deleting a run saves its settings and outcome summary atomically under `runs/model-lineage/archive/` before deletion; these archived steps remain in the graph and lose their live result-page link. This history snapshot does not preserve model weights or artifacts that the user explicitly deletes. If the history snapshot cannot be written, deletion fails without removing the run.

Names of previously saved checkpoints remain reserved after their files are removed. Choose a fresh output name for a new experiment; restoring the original files is a separate operation. This prevents old results from appearing to describe a different model later saved at the same path.

Live validation on September 24, 2026 followed `Qwen2.5-0.5B → direction #12 → surgery #14 → q05-beta0 → analysis #16`, then branched from analysis #16, changed the prompt, and completed analysis #17 on CUDA. Both analyses remain present with a recorded **forked from** connection. The 3B custom checkpoint `q3-beta0` also completed a baseline comparison as analysis #18. These are short functional checks, not evidence that the edits improved model behavior.

To export this workspace's history for another server without merging databases:

```bash
venv/bin/python scripts/export_model_lineage.py \
  --remote-root /home/ubuntu/aiasylum \
  --out runs/model-lineage/export.json
```

Copy that JSON into the destination's `runs/model-lineage/imports/` with a unique filename. The export includes an origin ID, exact model-reference mapping, manifests, and recorded run settings/results. Weight artifacts listed in the export may be copied separately under its `artifact_root`. This export does not load weights or rerun experiments.

Exports include archived runs and remap local branch IDs and model references for the destination. Previously imported bundles retain their separate origins: copy those original bundles separately and update their explicit model-reference mappings, following the export warnings. They are not flattened into this workspace's run registry.

## Transfer existing edited checkpoints to the GPU server

The GPU WebUI lists checkpoints available **on its API server**. A checkpoint saved in the workstation's `models/` directory is not usable by the remote API until its files have been copied there. Source deployment deliberately excludes `models/`; use the explicit model transfer command:

```bash
# Validate and transfer every local checkpoint:
scripts/remote_models.sh all

# Or transfer selected checkpoints:
scripts/remote_models.sh q05-beta0 q05-beta2 q3-rank2

# Inspect local readiness, fingerprints, and remote capacity first:
scripts/remote_models.sh --dry-run all
```

The default destination is `aiasylum-gpu:~/aiasylum/models/`. `--host SSH_ALIAS` and `--remote-dir PROJECT_DIR` select another configured SSH host or project directory inside that host's home. Transfers use existing SSH key authentication and do not send local API credentials.

Before uploading, the command requires nonempty configuration, tokenizer, weights, and `asylum_surgery.json` provenance. Sharded checkpoints must have every file referenced by the weight index. It hashes every file with SHA-256, checks remote free disk with a 10 GiB reserve, then transfers one checkpoint at a time using resumable `rsync`.

Partial uploads remain under `~/aiasylum/.model-transfers/staging/`, outside the model catalog. Only after the remote file list, byte sizes, and hashes match does the command atomically publish the checkpoint under `models/`. Local originals are retained. Existing identical checkpoints are verified and skipped; existing different checkpoints with the same name stop the transfer rather than being overwritten. No delete option is used.

Rerun the same command after an interrupted upload to resume it. A remote transfer lock prevents two transfer commands from publishing concurrently. If an SSH failure leaves `~/aiasylum/.model-transfers/transfer.lock`, first confirm no transfer process is active; then remove that stale lock directory and rerun.

Each execution records the full source manifest and one verified result per checkpoint under the ignored local directory `runs/model-transfers/<UTC-time>-<pid>/`. Matching source manifests and helper scripts are kept in the server's `.model-transfers/` directory. Publishing a model does not itself load it onto the GPU or validate its behavior; run a comparison or interpretability job separately.

### Correct originals using a private reconstruction

On a slow uplink, a saved direction can recreate most checkpoint bytes on the server's CPU. `scripts/seed_model_transfer.py` writes these **unverified private bases** only under `.model-transfers/`. CPU rounding or serialization can differ, so a rebuilt checkpoint is never silently substituted for an original.

A transfer audit includes `metadata.json` (original configuration and surgery provenance). After copying the needed saved direction directories, run the helper on the GPU server:

```bash
cd ~/aiasylum
venv/bin/python scripts/seed_model_transfer.py \
  --manifest .model-transfers/TRANSFER_ID/manifest.json \
  --metadata .model-transfers/TRANSFER_ID/metadata.json \
  --directions PATH_TO_SAVED_WEIGHT_RUNS \
  --output .model-transfers/bases/TRANSFER_ID \
  --models q05-beta0 q05-beta2
```

Then, from the workstation:

```bash
scripts/remote_models.sh --basis-root .model-transfers/bases/TRANSFER_ID q05-beta0 q05-beta2
```

The sender uses those files as a delta basis, corrects numerical differences, and still requires **every original file's exact SHA-256** before publication. Tiny rsync blocks reduce uploads caused by sparse rounding differences. GNU rsync is preferred automatically when available through Homebrew; `RSYNC_BIN` can select another executable. The helper checks the saved direction's source model and split before reconstructing, records basis receipts, and supports historical edits whose original weight files match a sibling with known provenance. Once one original is verified, its identical sibling weights can be reused without regenerating or retransmitting them.

When a slow connection makes even rsync's fixed-size checksums expensive, `scripts/repair_model_transfer.py` narrows differences with SHA-256 block hashes at 1 MiB, 64 KiB, 4 KiB, 256-byte, and 16-byte granularity. It sends the final changed blocks over a dedicated SSH connection, writes a separate private copy, and requires the complete original file's SHA-256 before making that copy available as a transfer basis:

```bash
venv/bin/python scripts/repair_model_transfer.py \
  --manifest runs/model-transfers/TRANSFER_ID/manifest.json \
  --basis-root .model-transfers/bases/TRANSFER_ID \
  --output-root .model-transfers/corrected/TRANSFER_ID \
  --report runs/model-transfers/repair-report.json \
  --models q3-rank2 q3-beta-neg2 q3-rank18

scripts/remote_models.sh \
  --basis-root .model-transfers/corrected/TRANSFER_ID \
  q3-rank2 q3-beta-neg2 q3-rank18
```

This helper never writes to the visible `models/` directory. A wrong file length, incomplete correction, or final hash mismatch stops the operation. Local originals and the initial reconstruction are preserved, and the correction report records hashes, transfer bytes, and verified files.

## Custom checkpoints on AWS

All custom checkpoints live on `aiasylum-gpu` under `/home/ubuntu/aiasylum/models/`. The workstation's `models/` directory is intentionally empty.

On **2026-09-25 at 15:55 UTC**, every custom checkpoint was deleted on both machines: nine on the workstation and ten on AWS. Records are in `runs/model-deletions/`. Neither machine had a backup, so the original weight files are gone. The old names now show as **deleted** in the library and keep their history.

The same day, all ten were rebuilt on AWS as `<name>-rebuild`: `ablated-rebuild`, `ablated_r2-rebuild`, `q05-beta0-rebuild`, `q05-beta2-rebuild`, `q3-beta-neg2-rebuild`, `q3-beta0-rebuild`, `q3-rank18-rebuild`, `q3-rank2-rebuild`, `qwen0.5-ablated-rebuild` and `qwen05b-control-rebuild`. Each one replays its recorded surgery on CPU, with the same base model, saved direction, beta/rank/k and dtype.

These are rebuilds, not the originals:

- `qwen05b-control-rebuild` matches the original's SHA-256 exactly.
- `qwen0.5-ablated-rebuild` reproduces the original's recorded mean relative change to every digit, but its original hash was never recorded.
- The other eight have the same file sizes but differ at rounding level. Their mean relative change is within 0.05% of the original's.

Each `asylum_surgery.json` records these checks under `extra.rebuild`. Results recorded against the old names describe the deleted files, not the rebuilds.

The rebuild script and its log are on the server at `.model-transfers/rebuild_checkpoints.py` and `.model-transfers/rebuild-20260925.log`. The recipes and original SHA-256 manifest it read from are in `.model-transfers/completed/20260924-originals/`. The workstation copy of that manifest is `runs/model-transfers/20260924T200926Z-32676/manifest.json`.

A 10-question MMLU comparison of `q05-beta0-rebuild` against `Qwen/Qwen2.5-0.5B-Instruct` (campaign `4f5183e55b6b4aa6bd88486ae0057de3`) completed with 2/10 for each and no invalid or truncated answers. That checks the workflow, not the effect of the edit.
