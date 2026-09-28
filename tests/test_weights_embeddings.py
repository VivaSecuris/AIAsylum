"""Embedding cartography: alignment maps, anchors, and black-box dimension recovery."""

import pytest

torch = pytest.importorskip("torch")

from vivasecuris.aiasylum.weights.embeddings import (
    LocalLogitOracle,
    align_from_matrices,
    procrustes,
    recover_dimension,
    recover_subspace,
    relative_agreement,
    relative_repr,
    retrieval_at_k,
    score_recovery,
    shared_anchors,
)


def _random_orthogonal(d, seed=0):
    torch.manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))
    return q


# --- alignment maths --------------------------------------------------------

def test_procrustes_recovers_a_known_rotation():
    torch.manual_seed(1)
    X = torch.randn(200, 16, dtype=torch.float64)
    R = _random_orthogonal(16, seed=2)
    Y = X @ R
    M = procrustes(X, Y)
    assert torch.allclose(X @ M, Y, atol=1e-6)
    assert torch.allclose(M, R, atol=1e-5)


def test_self_alignment_is_perfect_retrieval():
    torch.manual_seed(3)
    E = torch.randn(60, 16)
    ids = list(range(60))
    res = align_from_matrices(E, E, ids, ids, method="procrustes", seed=0)
    assert res.retrieval[1] == 1.0


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")),
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
])
@pytest.mark.parametrize("different_width", [False, True])
def test_alignment_supports_accelerator_and_mixed_device_embeddings(device, different_width):
    """The estimators agree with CPU for rotations and unequal-width maps.

    Include an optional random-init null and relative coordinates: both used
    to try unsupported float64 operations on live MPS weights as well.
    """
    from vivasecuris.aiasylum.weights.embeddings import relative_agreement_with_null

    g = torch.Generator().manual_seed(31)
    a = torch.randn(80, 8, generator=g)
    b = a @ torch.randn(8, 12, generator=g) if different_width else a.clone()
    random_b = torch.randn(b.shape, generator=g)
    ids = list(range(60))
    expected = align_from_matrices(a, b, ids, ids, random_init_E_b=random_b)
    for b_device in ("cpu", device):
        actual = align_from_matrices(a.to(device), b.to(b_device), ids, ids,
                                     random_init_E_b=random_b.to(device))
        assert actual.retrieval == expected.retrieval
        assert actual.null_shuffled == expected.null_shuffled
        assert actual.null_random_init == expected.null_random_init
        assert actual.cosine == pytest.approx(expected.cosine, abs=1e-6)
        assert relative_agreement_with_null(a.to(device), b.to(b_device), ids, ids) == pytest.approx(
            relative_agreement_with_null(a, b, ids, ids), abs=1e-6)


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")),
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
])
def test_recovery_scores_live_accelerator_weights_on_cpu(device):
    g = torch.Generator().manual_seed(32)
    weights = torch.randn(24, 4, generator=g)
    logits = torch.randn(80, 4, generator=g) @ weights.T
    d, _ = recover_dimension(logits.to(device))
    assert d == 4
    recovered = recover_subspace(logits.to(device), d)
    assert recovered.device.type == "cpu" and recovered.dtype == torch.float64
    assert score_recovery(recovered, weights.to(device))["subspace_overlap"] > 0.999


def test_shuffled_anchor_null_collapses():
    torch.manual_seed(4)
    E = torch.randn(80, 16)
    ids = list(range(80))
    res = align_from_matrices(E, E, ids, ids, method="procrustes", seed=0)
    # Real alignment is perfect; shuffling the training correspondence destroys it.
    assert res.null_shuffled[1] < res.retrieval[1]
    assert res.null_shuffled[1] <= 0.5


def test_relative_repr_is_rotation_invariant():
    torch.manual_seed(5)
    E = torch.randn(40, 12)
    R = _random_orthogonal(12, seed=6).float()
    anchors = E[torch.arange(0, 40, 4)]
    rel_a = relative_repr(E, anchors)
    rel_b = relative_repr(E @ R, anchors @ R)
    assert torch.allclose(rel_a, rel_b, atol=1e-6)


def test_relative_agreement_high_for_rotation_low_for_random():
    torch.manual_seed(7)
    E = torch.randn(50, 16)
    R = _random_orthogonal(16, seed=8).float()
    ids = list(range(50))
    same = relative_agreement(E, E @ R, ids, ids)
    other = relative_agreement(E, torch.randn(50, 16), ids, ids)
    assert same > 0.99
    assert other < same


def test_retrieval_at_k_basic():
    # Query i should retrieve target i when they are identical rows.
    T = torch.randn(10, 8)
    r = retrieval_at_k(T, T, list(range(10)), ks=(1, 3))
    assert r[1] == 1.0 and r[3] == 1.0


def test_retrieval_searches_every_row_not_just_anchors():
    """A near-duplicate distractor elsewhere in the vocab must be able to win.

    Guards the double-check fix: retrieval restricted to the anchor pool inflated
    P@1 (99.6% vs 86.4% over the full gpt2-medium vocab).
    """
    torch.manual_seed(11)
    T = torch.randn(10, 8)
    D = T + 0.01 * torch.randn(10, 8)                 # distractors, rows 10..19
    full = torch.cat([T, D])
    # Queries sit exactly on the distractors; the "correct" rows are the T rows.
    assert retrieval_at_k(D, T, list(range(10)))[1] == 1.0      # anchor-only pool
    assert retrieval_at_k(D, full, list(range(10)))[1] == 0.0   # full pool


def test_align_reports_full_vocab_pool():
    torch.manual_seed(12)
    E = torch.randn(70, 16)
    res = align_from_matrices(E, E, list(range(40)), list(range(40)), method="procrustes")
    assert res.extra["retrieval_pool"] == 70
    assert res.retrieval[1] == 1.0                    # still exact with no duplicate rows


def test_relative_agreement_centering_defeats_anisotropy():
    """Unrelated spaces sharing a big common offset look alike uncentered.

    Guards the double-check fix: on gpt2 vs gpt2-medium the uncentered score was
    0.994 against a 0.979 shuffled null (uninformative); centered it separates.
    """
    from vivasecuris.aiasylum.weights.embeddings import relative_agreement_with_null

    torch.manual_seed(13)
    n, d = 60, 16
    A = torch.randn(n, d) + 20.0 * torch.ones(d)      # anisotropic: shared direction
    B = torch.randn(n, d) + 20.0 * torch.ones(d)      # unrelated to A, same anisotropy
    ids = list(range(n))
    unc = relative_agreement_with_null(A, B, ids, ids, center=False)
    cen = relative_agreement_with_null(A, B, ids, ids, center=True)
    assert unc["real"] > 0.95                         # the anisotropy trap
    assert cen["real"] < 0.5                          # centered: unrelated reads as unrelated

    R = _random_orthogonal(d, seed=14).float()
    rot = relative_agreement_with_null(A, A @ R, ids, ids, center=True)
    assert rot["real"] > 0.99 and rot["null"] < rot["real"] - 0.3


# --- anchors ----------------------------------------------------------------

def test_shared_anchors_are_mutual_single_tokens():
    from tests._tiny_lm import build_tokenizer

    from vivasecuris.aiasylum.weights.remap import single_token_id

    tok_a = build_tokenizer(words=["[UNK]", "[PAD]", "[EOS]", "alpha", "beta", "gamma", "onlyA"])
    tok_b = build_tokenizer(words=["[UNK]", "[PAD]", "[EOS]", "beta", "gamma", "delta", "onlyB"])
    ids_a, ids_b, strings = shared_anchors(
        tok_a, tok_b, extra_words=["alpha", "beta", "gamma", "delta", "onlyA", "onlyB"],
    )
    # Only tokens single in BOTH survive.
    assert set(strings) == {"beta", "gamma"}
    for s, ia, ib in zip(strings, ids_a, ids_b):
        assert single_token_id(tok_a, s) == ia
        assert single_token_id(tok_b, s) == ib


# --- black-box dimension / subspace recovery --------------------------------

def test_recover_dimension_and_subspace_from_synthetic_logits():
    torch.manual_seed(9)
    N, V, d = 200, 60, 6
    H = torch.randn(N, d, dtype=torch.float64)
    W_U = torch.randn(V, d, dtype=torch.float64)       # [V, d]
    b = torch.randn(V, dtype=torch.float64)
    L = H @ W_U.t() + b                                 # [N, V]

    rec_d, spectrum = recover_dimension(L, max_dim=20)
    assert rec_d == d
    # spectrum drops sharply after d
    assert spectrum[d - 1] > 1e3 * spectrum[d] if spectrum[d] > 0 else True

    basis = recover_subspace(L, d)                      # [d, V]
    score = score_recovery(basis, W_U)
    assert score["subspace_overlap"] > 0.999
    assert score["max_angle_deg"] < 1.0


def test_recover_subspace_with_column_subset():
    torch.manual_seed(10)
    N, V, d = 150, 80, 5
    H = torch.randn(N, d, dtype=torch.float64)
    W_U = torch.randn(V, d, dtype=torch.float64)
    L = H @ W_U.t() + torch.randn(V, dtype=torch.float64)
    cols = torch.randperm(V)[:40].sort().values
    Lsub = L[:, cols]
    rec_d, _ = recover_dimension(Lsub, max_dim=20)
    assert rec_d == d
    basis = recover_subspace(Lsub, d)
    score = score_recovery(basis, W_U, cols=cols)
    assert score["subspace_overlap"] > 0.999


# --- guardrail --------------------------------------------------------------

def test_oracle_rejects_non_local_model():
    with pytest.raises(ValueError, match="Ollama"):
        LocalLogitOracle(model=None, tokenizer=None, model_id="llama3:8b")
