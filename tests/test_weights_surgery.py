"""Property tests for the direction-scaling weight edit.

These run on synthetic tensors with no model download, and they are the guard
against the failure mode that matters most here: a transposed projection that
looks plausible, runs without error, and silently does nothing (or destroys the
whole matrix instead of one direction).
"""

import pytest

torch = pytest.importorskip("torch")

from vivasecuris.aiasylum.interp.core.arch import KIND_EMBED, KIND_OUT
from vivasecuris.aiasylum.weights.surgery import scale_direction_component

D_MODEL, D_IN, VOCAB = 64, 128, 256


@pytest.fixture
def rng():
    torch.manual_seed(0)


@pytest.fixture
def direction():
    torch.manual_seed(1)
    r = torch.randn(D_MODEL)
    return r / r.norm()


def test_out_kind_ablation_removes_exactly_one_direction(rng, direction):
    """beta=0 must zero the r-component of the output and keep the rest."""
    W = torch.randn(D_MODEL, D_IN)
    W_abl = scale_direction_component(W, direction, beta=0.0, kind=KIND_OUT)

    x = torch.randn(32, D_IN)
    y, y_abl = x @ W.T, x @ W_abl.T

    # The direction is gone.
    assert (y_abl @ direction).abs().max() < 1e-4

    # But the signal is not: only one of d_model directions was removed.
    assert y_abl.norm() > 0.8 * y.norm()

    # And the orthogonal complement is untouched.
    y_perp = y - torch.outer(y @ direction, direction)
    assert torch.allclose(y_abl, y_perp, atol=1e-4)


def test_beta_one_is_identity(rng, direction):
    """beta=1 must be a no-op, or every downstream delta is measuring noise."""
    for kind, shape in ((KIND_OUT, (D_MODEL, D_IN)), (KIND_EMBED, (VOCAB, D_MODEL))):
        W = torch.randn(*shape)
        out = scale_direction_component(W, direction, beta=1.0, kind=kind)
        assert torch.allclose(W, out, atol=1e-6), f"beta=1 changed {kind}"


@pytest.mark.parametrize("beta", [0.0, 0.5, 2.0, 3.0])
def test_out_kind_scales_component_by_beta(rng, direction, beta):
    """The r-component scales by exactly beta; the complement never moves."""
    W = torch.randn(D_MODEL, D_IN)
    W_new = scale_direction_component(W, direction, beta=beta, kind=KIND_OUT)

    x = torch.randn(32, D_IN)
    y, y_new = x @ W.T, x @ W_new.T

    assert torch.allclose(y_new @ direction, beta * (y @ direction), atol=1e-4)

    perp = lambda v: v - torch.outer(v @ direction, direction)
    assert torch.allclose(perp(y_new), perp(y), atol=1e-4)


def test_embed_kind_ablates_every_row(rng, direction):
    """For the embedding table each row is a residual vector in its own right."""
    W = torch.randn(VOCAB, D_MODEL)
    W_abl = scale_direction_component(W, direction, beta=0.0, kind=KIND_EMBED)

    assert (W_abl @ direction).abs().max() < 1e-4
    assert W_abl.norm() > 0.8 * W.norm()


def test_non_unit_direction_is_normalized(rng):
    """A caller passing an unnormalized direction must not get a scaled edit."""
    torch.manual_seed(2)
    r = torch.randn(D_MODEL)
    W = torch.randn(D_MODEL, D_IN)

    a = scale_direction_component(W, r, beta=0.0, kind=KIND_OUT)
    b = scale_direction_component(W, r * 7.3, beta=0.0, kind=KIND_OUT)
    assert torch.allclose(a, b, atol=1e-5)


def test_bfloat16_roundtrip_still_ablates(rng, direction):
    """bf16 is the real storage dtype; the edit must survive the cast back."""
    W = torch.randn(D_MODEL, D_IN).to(torch.bfloat16)
    W_abl = scale_direction_component(W, direction, beta=0.0, kind=KIND_OUT)

    assert W_abl.dtype == torch.bfloat16
    x = torch.randn(32, D_IN).to(torch.bfloat16)
    residual = ((x @ W_abl.T).to(torch.float32) @ direction).abs().max()
    # bf16 has ~3 decimal digits, so this is the honest tolerance.
    assert residual < 0.5, f"direction survived ablation in bf16: {residual}"


def test_shape_mismatch_raises_rather_than_broadcasting(rng):
    """A wrong-sized direction must fail loudly, not broadcast into nonsense."""
    W = torch.randn(D_MODEL, D_IN)
    wrong = torch.randn(D_IN)  # matches the input dim, not the residual dim
    with pytest.raises(ValueError, match="Shape mismatch"):
        scale_direction_component(W, wrong, beta=0.0, kind=KIND_OUT)


def test_unknown_kind_raises(rng, direction):
    W = torch.randn(D_MODEL, D_IN)
    with pytest.raises(ValueError, match="Unknown matrix kind"):
        scale_direction_component(W, direction, beta=0.0, kind="sideways")


# --- subspace edits -------------------------------------------------------

from vivasecuris.aiasylum.weights.surgery import remove_subspace_component


def _ortho_basis(m, d, seed):
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, m))  # [d, m], orthonormal columns
    return Q.t().contiguous()                  # [m, d], orthonormal rows


def test_rank1_subspace_equals_beta0(direction):
    """A rank-1 subspace at k=1 must reproduce a beta=0 single-direction ablation."""
    for kind, shape in ((KIND_OUT, (D_MODEL, D_IN)), (KIND_EMBED, (VOCAB, D_MODEL))):
        torch.manual_seed(7)
        W = torch.randn(*shape)
        a = remove_subspace_component(W, direction.reshape(1, -1), k=1.0, kind=kind)
        b = scale_direction_component(W, direction, beta=0.0, kind=kind)
        assert torch.allclose(a, b, atol=1e-5), f"rank-1 k=1 != beta=0 for {kind}"


def test_subspace_removes_every_basis_direction(rng):
    """k=1 removes the whole subspace: no basis component survives, signal does."""
    B = _ortho_basis(4, D_MODEL, 3)
    W = torch.randn(D_MODEL, D_IN)
    W2 = remove_subspace_component(W, B, k=1.0, kind=KIND_OUT)

    x = torch.randn(32, D_IN)
    y, y2 = x @ W.T, x @ W2.T
    for j in range(B.shape[0]):
        assert (y2 @ B[j]).abs().max() < 1e-4, f"basis row {j} survived"
    # Only 4 of D_MODEL directions removed; most of the signal remains.
    assert y2.norm() > 0.8 * y.norm()


def test_subspace_k_scales_component(rng):
    """The component along the subspace scales by (1 - k); complement is fixed."""
    B = _ortho_basis(3, D_MODEL, 4)
    W = torch.randn(D_MODEL, D_IN)
    x = torch.randn(32, D_IN)
    y = x @ W.T
    for k in (0.0, 0.5, 1.0, 1.5, 2.0):
        y2 = x @ remove_subspace_component(W, B, k=k, kind=KIND_OUT).T
        for j in range(B.shape[0]):
            assert torch.allclose(y2 @ B[j], (1.0 - k) * (y @ B[j]), atol=1e-4)


def test_subspace_leaves_orthogonal_complement_untouched(rng):
    """A residual direction outside the subspace must not move."""
    B = _ortho_basis(3, D_MODEL, 5)
    W = torch.randn(D_MODEL, D_IN)
    v = torch.randn(D_MODEL)
    u = v - B.t() @ (B @ v)          # project v into the complement of the subspace
    u = u / u.norm()

    x = torch.randn(32, D_IN)
    y = x @ W.T
    y2 = x @ remove_subspace_component(W, B, k=1.0, kind=KIND_OUT).T
    assert torch.allclose(y2 @ u, y @ u, atol=1e-4)


def test_subspace_embed_ablates_every_row(rng):
    B = _ortho_basis(4, D_MODEL, 6)
    W = torch.randn(VOCAB, D_MODEL)
    W2 = remove_subspace_component(W, B, k=1.0, kind=KIND_EMBED)
    for j in range(B.shape[0]):
        assert (W2 @ B[j]).abs().max() < 1e-4
    assert W2.norm() > 0.8 * W.norm()


def test_subspace_shape_mismatch_raises(rng):
    W = torch.randn(D_MODEL, D_IN)
    wrong = _ortho_basis(2, D_IN, 8)  # rows sized to the input dim, not the residual dim
    with pytest.raises(ValueError, match="Shape mismatch"):
        remove_subspace_component(W, wrong, k=1.0, kind=KIND_OUT)
