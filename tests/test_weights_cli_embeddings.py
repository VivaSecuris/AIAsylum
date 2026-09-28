"""Embedding reconstruction's CLI path on real tiny-model logits and geometry."""

import json

import pytest
from click.testing import CliRunner

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tests._tiny_lm import build_model, build_tokenizer
from vivasecuris.aiasylum.weights import cli as cli_mod
from vivasecuris.aiasylum.weights import embeddings as embeddings_mod


@pytest.mark.parametrize("extra_cols,full_vocab", [(0, False), (64, True)])
def test_embed_recon_cli_handles_explicit_and_implicit_vocabulary_columns(
    monkeypatch, tmp_path, extra_cols, full_vocab,
):
    """Exercise real collection, recovery and alignment for both column contracts.

    Check the exact anchor rows received by alignment, so a fix that avoids the
    None crash but silently maps token IDs to the wrong recovered rows also fails.
    """
    words = ["[UNK]", "[PAD]", "[EOS]"] + list(dict.fromkeys(embeddings_mod._WORD_BANK))[:61]
    tok = build_tokenizer(words=words)
    model = build_model()
    monkeypatch.setattr(cli_mod, "_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr("vivasecuris.aiasylum.interp.core.loader.load",
                        lambda *args, **kwargs: (model, tok))
    seen = {}
    original_collect = embeddings_mod.collect_logits
    original_recover = embeddings_mod.recover_token_embeddings
    original_align = embeddings_mod.align_from_matrices

    def collect(*args, **kwargs):
        logits, cols = original_collect(*args, **kwargs)
        seen.update(cols=cols, anchor_ids=list(kwargs["must_include"]), vocab=logits.shape[1])
        return logits, cols

    def recover(*args, **kwargs):
        result = original_recover(*args, **kwargs)
        seen["recovered"] = result
        return result

    def align(recovered, reference, ids_a, ids_b, **kwargs):
        cols = list(range(seen["vocab"])) if seen["cols"] is None else seen["cols"].tolist()
        positions = {token_id: row for row, token_id in enumerate(cols)}
        expected = torch.stack([seen["recovered"][positions[token_id]] for token_id in seen["anchor_ids"]])
        assert torch.equal(recovered, expected)
        assert ids_a == list(range(len(seen["anchor_ids"])))
        assert ids_b == seen["anchor_ids"]  # the two models share a tokenizer
        seen["aligned"] = True
        return original_align(recovered, reference, ids_a, ids_b, **kwargs)

    monkeypatch.setattr(embeddings_mod, "collect_logits", collect)
    monkeypatch.setattr(embeddings_mod, "recover_token_embeddings", recover)
    monkeypatch.setattr(embeddings_mod, "align_from_matrices", align)
    report = tmp_path / "reconstruction.json"
    result = CliRunner().invoke(cli_mod.weights, [
        "embed-recon", "--oracle", "local-oracle", "--reference", "local-reference",
        "--n-queries", "80", "--max-anchors", "24", "--extra-cols", str(extra_cols),
        "--device", "cpu", "--dtype", "float32", "--out", str(report),
    ])
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert seen["aligned"]
    assert (seen["cols"] is None) is full_vocab
    assert seen["vocab"] == (64 if full_vocab else 24)
    summary = json.loads(report.read_text())
    assert summary["n_anchors"] == 24 and summary["extra"]["n_anchor_cols"] == 24
    assert 0 < summary["extra"]["recovered_dim"] <= 32
    assert summary["extra"]["retrieval_pool"] == 64
    assert set(summary["retrieval"]) == {"1", "5"}
    assert "aligned 24 anchors" in result.output
