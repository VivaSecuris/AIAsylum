"""H-Neurons: CETT math, L1 selection, MoE refusal, and hook==baked equivalence."""

import pytest

torch = pytest.importorskip("torch")

from tests._tiny_lm import build_model, build_tokenizer
from vivasecuris.aiasylum.weights.hneurons import (
    HNeuronSet,
    bake_hneurons,
    cett_per_token,
    column_norms,
    scale_neurons,
    select_hneurons,
    _dense_down_projs,
)


def test_column_norms_shape_and_values():
    m = build_model()
    _, down = _dense_down_projs(m)[0]
    cn = column_norms(down)
    assert cn.shape[0] == down.weight.shape[1]
    # column j norm == norm of W[:, j]
    assert torch.allclose(cn[0], torch.linalg.vector_norm(down.weight[:, 0].float()))


def test_cett_hand_calculation():
    # One token, d_ff=2, d_model=2. col_norm=[3,4], z=[2,-1], out=[3,4] -> ||out||=5.
    z = torch.tensor([[2.0, -1.0]])
    out = torch.tensor([[3.0, 4.0]])
    col_norm = torch.tensor([3.0, 4.0])
    cett = cett_per_token(z, out, col_norm)
    # |2|*3/5 = 1.2 ; |−1|*4/5 = 0.8
    assert torch.allclose(cett, torch.tensor([[1.2, 0.8]]), atol=1e-6)


def test_select_recovers_a_planted_neuron():
    import numpy as np

    rng = np.random.default_rng(0)
    n, n_layers, d_ff = 200, 2, 4
    F = 2 * n_layers * d_ff
    feature_map = [(L, j) for L in range(n_layers) for j in range(d_ff) for _ in (0, 1)]
    # Column order: for each (layer, neuron) two adjacent columns (answer, other).
    feature_map = []
    for L in range(n_layers):
        for j in range(d_ff):
            feature_map.append((L, j))   # answer feature
            feature_map.append((L, j))   # other feature
    y = rng.integers(0, 2, size=n)
    X = rng.normal(size=(n, F)) * 0.01
    planted_col = 2 * (0 * d_ff + 1)     # layer 0, neuron 1, answer feature
    X[:, planted_col] += y * 3.0         # strongly correlated with label
    hset = select_hneurons(X, y, feature_map, d_ff=d_ff, top_k=F, c_grid=(0.1, 1.0), seed=0)
    assert 1 in hset.neurons.get(0, [])
    assert hset.n_total == n_layers * d_ff
    assert hset.auroc > 0.7


def test_moe_model_is_refused():
    moe = build_model(family="qwen2_moe")
    with pytest.raises(ValueError, match="mixture-of-experts"):
        _dense_down_projs(moe)


def test_hook_equals_baked_logits():
    """The core guarantee: scaling down_proj input == scaling W_down columns."""
    torch.manual_seed(0)
    m = build_model()
    tok = build_tokenizer()
    ids = tok(["w3 w4 w5 w6"], return_tensors="pt")

    # Select a couple of neurons in each layer.
    down0 = _dense_down_projs(m)[0][1]
    d_ff = down0.weight.shape[1]
    hset = HNeuronSet(neurons={0: [1, 3], 1: [2]}, model_id="tiny", d_ff=d_ff,
                      n_selected=3, n_total=2 * d_ff)
    alpha = 0.4

    with torch.no_grad():
        with scale_neurons(m, hset, alpha):
            hooked = m(**ids).logits.clone()

    # Bake the same scale into a fresh copy and compare.
    m2 = build_model()      # identical seed -> identical weights
    bake_hneurons(m2, hset, alpha)
    with torch.no_grad():
        baked = m2(**ids).logits

    assert torch.allclose(hooked, baked, atol=1e-5), (hooked - baked).abs().max()


def test_bake_reports_summary_fields():
    """bake_hneurons returns the fields save_edited_model's summary needs."""
    m = build_model()
    d_ff = _dense_down_projs(m)[0][1].weight.shape[1]
    hset = HNeuronSet(neurons={0: [1, 2], 1: [3]}, model_id="tiny", d_ff=d_ff,
                      n_selected=3, n_total=2 * d_ff)
    out = bake_hneurons(m, hset, 0.5)
    assert out["matrices_edited"] == 2          # two layers had neurons
    assert out["neurons_scaled"] == 3
    assert out["mean_relative_change"] >= 0.0
    # alpha=1 is a no-op with zero relative change.
    m2 = build_model()
    assert bake_hneurons(m2, hset, 1.0)["mean_relative_change"] == 0.0


def test_capture_cett_shapes():
    from vivasecuris.aiasylum.weights.hneurons import capture_cett

    m, tok = build_model(), build_tokenizer()
    d_ff = _dense_down_projs(m)[0][1].weight.shape[1]
    n_layers = len(_dense_down_projs(m))
    feats, fmap, dff, alens = capture_cett(m, tok, ["w3 w4", "w5 w6 w7"], max_new_tokens=4)
    assert feats.shape == (2, 2 * n_layers * d_ff)
    assert dff == d_ff and len(fmap) == 2 * n_layers * d_ff
    assert len(alens) == 2
    # feature_map columns map to valid (layer, neuron) pairs
    assert all(0 <= j < d_ff for _, j in fmap)


def test_label_consistency_partitions():
    from vivasecuris.aiasylum.weights.hneurons import label_consistency

    m, tok = build_model(), build_tokenizer()
    rows = [{"question": "w3 w4", "aliases": ["w10"]},
            {"question": "w5 w6", "aliases": ["w11"]}]
    out = label_consistency(m, tok, rows, n_samples=3, max_new_tokens=4, seed=0)
    # Every row lands in at most one bucket; buckets are disjoint subsets of input.
    assert set(out) == {"correct", "incorrect"}
    assert len(out["correct"]) + len(out["incorrect"]) <= len(rows)


def test_triviaqa_standardization():
    from vivasecuris.aiasylum.benchmarks.datasets import standardize_benchmark_row

    item = {
        "question": "What is the capital of France?",
        "answer": {"value": "Paris", "aliases": ["Paris, France"],
                   "normalized_aliases": ["paris"]},
    }
    row = standardize_benchmark_row("triviaqa", {"question_field": "question", "answer_field": "answer"}, item)
    assert row["answer"] == "Paris"
    assert "Paris" in row["aliases"] and "paris" in row["aliases"]
    # canonical value is first and there are no duplicates
    assert row["aliases"][0] == "Paris"
    assert len(row["aliases"]) == len(set(row["aliases"]))


def test_hneuronset_roundtrip(tmp_path):
    hset = HNeuronSet(neurons={0: [1, 3], 2: [5]}, model_id="tiny", d_ff=64,
                      auroc=0.78, null_auroc_p95=0.6, n_selected=3, n_total=128)
    hset.save(tmp_path)
    back = HNeuronSet.load(tmp_path)
    assert back.neurons == {0: [1, 3], 2: [5]}
    assert back.auroc == pytest.approx(0.78)
    assert back.beats_null


@pytest.mark.parametrize("scores,labels,expected", [
    ([1] * 10, [0, 1] * 5, 0.5),
    ([0, 1, 1, 2], [0, 0, 1, 1], 0.875),
    ([0, 1, 1, 2], [1, 1, 0, 0], 0.125),
])
def test_auroc_gives_tied_pairs_half_credit(scores, labels, expected):
    from vivasecuris.aiasylum.weights.hneurons import _auroc

    assert _auroc(scores, labels) == pytest.approx(expected)
    assert _auroc(scores[::-1], labels[::-1]) == pytest.approx(expected)


def test_noise_only_review_reproduction_does_not_clear_null():
    """The review's 80 x 10,000 independent-noise detector used to score 0.889."""
    import numpy as np

    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 10000))
    y = rng.integers(0, 2, size=80)
    fmap = [(0, j // 2) for j in range(X.shape[1])]
    hset = select_hneurons(X, y, fmap, d_ff=5000, top_k=64, c_grid=(1.0,), seed=0,
                          answer_lengths=np.ones(80))
    assert hset.auroc < 0.75
    assert not hset.beats_null
    assert not hset.usable
    assert hset.surface_auroc == 0.5


def test_report_labels_cannot_change_selected_neurons_or_c():
    import numpy as np

    rng = np.random.default_rng(23)
    X = rng.normal(size=(100, 100))
    y = rng.integers(0, 2, size=100)
    X[:, 0] += y * 1.5
    fmap = [(0, j // 2) for j in range(X.shape[1])]
    kwargs = dict(d_ff=50, top_k=16, c_grid=(0.01, 0.1, 1.0), seed=0)
    before = select_hneurons(X, y, fmap, **kwargs)
    changed = y.copy()
    changed[before.extra["report_indices"]] = 1 - changed[before.extra["report_indices"]]
    after = select_hneurons(X, changed, fmap, **kwargs)
    assert before.neurons == after.neurons
    assert before.extra["C"] == after.extra["C"]
    assert before.extra["train_indices"] == after.extra["train_indices"]
    assert after.auroc == pytest.approx(1 - before.auroc)


def test_null_repeats_feature_selection_and_tuning(monkeypatch):
    import numpy as np
    from vivasecuris.aiasylum.weights import hneurons as hn

    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 40))
    y = np.array([0, 1] * 40)
    fitted = []
    original = hn._fit_selection

    def record(features, labels, top_k, c_grid, seed):
        result = original(features, labels, top_k, c_grid, seed)
        fitted.append((features.shape, labels.copy(), result[0].copy(), tuple(c_grid)))
        return result

    monkeypatch.setattr(hn, "_fit_selection", record)
    result = select_hneurons(X, y, [(0, j // 2) for j in range(40)],
                            d_ff=20, top_k=8, c_grid=(0.1, 1.0))
    assert len(fitted) == 21  # true labels and twenty independently selected nulls
    assert all(shape == (56, 40) and grid == (0.1, 1.0) for shape, _, _, grid in fitted)
    assert all(not np.array_equal(labels, fitted[0][1]) for _, labels, _, _ in fitted[1:])
    assert any(not np.array_equal(keep, fitted[0][2]) for _, _, keep, _ in fitted[1:])
    assert result.extra["null_full_selection"] is True


@pytest.mark.parametrize("surface,selected,expected", [
    (0.90, 2, False),
    (0.50, 2, True),
    (float("nan"), 2, False),
    (0.50, 0, False),
])
def test_usable_requires_selected_neurons_and_both_baselines(surface, selected, expected):
    hset = HNeuronSet(neurons={0: [1, 2]} if selected else {}, model_id="tiny", d_ff=4,
                     auroc=0.75, null_auroc_p95=0.55, surface_auroc=surface,
                     n_selected=selected, n_total=4)
    assert hset.beats_null
    assert hset.usable is expected
    assert hset.metadata()["usable"] is expected
    assert hset.metadata()["beats_surface"] is (surface == 0.50)


def test_missing_class_in_report_is_rejected():
    import numpy as np

    y = np.zeros(20, dtype=int)
    y[np.random.default_rng(0).permutation(20)[-1]] = 1
    with pytest.raises(ValueError, match="partitions must each contain both classes"):
        select_hneurons(np.ones((20, 4)), y, [(0, 0)] * 4, d_ff=2)
