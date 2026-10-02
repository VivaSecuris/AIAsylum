# Third-party notices

AI Asylum is proprietary software. See [LICENSE](LICENSE). The pieces below are included under their own terms and are not covered by that proprietary grant.

## Vendored interpretability engine

`vivasecuris/aiasylum/interp/` is vendored from the labotomy project (`llm_prompt_diff`), trimmed to the analysis core. Model loading was rewritten; see the module docstring in `vivasecuris/aiasylum/interp/__init__.py`. Labotomy is a VivaSecuris project and is redistributed here under the same proprietary terms as the rest of this repository.

## Jailbreak corpora (git submodules)

These directories are git submodules. Fetch them with `git clone --recursive` or `git submodule update --init`. Each keeps its upstream license.

| Path | Upstream | License |
|------|----------|---------|
| `docs/jailbreaks/LLM-Jailbreaks` | https://github.com/langgptai/LLM-Jailbreaks | Apache-2.0 (`LICENSE` in the submodule) |
| `docs/jailbreaks/Prompt-Hacking-Resources` | https://github.com/PromptLabs/Prompt-Hacking-Resources | MIT (`LICENSE` in the submodule) |
| `docs/jailbreaks/jailbreak_llms` | https://github.com/verazuo/jailbreak_llms | MIT (`LICENSE` in the submodule) |
| `docs/jailbreaks/Awesome-LLM-Jailbreak` | https://github.com/Meirtz/Awesome-LLM-Jailbreak | No license file in the pinned commit. Treat it as upstream's curated link list; do not assume a reuse grant beyond what that repository states. |
