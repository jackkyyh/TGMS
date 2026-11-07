import torch
from qutip import Qobj, rand_dm

from adaptive.utils import evaluate_backbone


def to_torch(qobjs):
    if not isinstance(qobjs, list):
        qobjs = [qobjs]
    if isinstance(qobjs[0], Qobj):
        qobjs = [qobj.full() for qobj in qobjs]
    return torch.stack([torch.from_numpy(qobj) for qobj in qobjs])


def check_unitary(U):
    assert torch.allclose(cast_real(U @ U.mH), torch.eye(U.shape[-1])), "Not unitary!"


def check_hermicity(U):
    assert torch.allclose(U, U.mH, atol=1e-5), f"Not hermitian! Diff: {(U - U.mH).abs().max()}"


def check_positive(rho):
    L, Q = torch.linalg.eigh(rho)
    assert L.min() >= -1e-6, f"Not positive semidefinite! {L.min()}"


def check_trace(rho):
    tr = trace(rho)
    assert torch.allclose(tr, torch.ones_like(tr)), f"Not trace 1! {tr.min(), tr.max()}"


def check_density(rho):
    check_positive(rho)
    check_trace(rho)


def cast_real(rho, tol=1e-4):
    assert rho.imag.abs().max() < tol, f"Not real! Max imag: {rho.imag.abs().max()}"
    return rho.real


def trace(rho, cast=True):
    tr = torch.einsum("...ii->...", rho)
    if cast:
        tr = cast_real(tr)
    return tr


def expect(m_ops, rho):
    if is_state(rho):
        exp = torch.einsum("bki, bcij,bjk->bck", rho.mH, m_ops, rho)
    else:
        exp = torch.einsum("bcij,bji->bc", m_ops, rho)
        exp = exp.unsqueeze(-1)

    exp = cast_real(exp, tol=1e-5)

    return exp


def sqrtm(rho):

    L, Q = torch.linalg.eigh(rho)
    L[L < 0] = 0
    L = torch.diag_embed(L.sqrt())
    sqrt_rho = Q @ (L + 0j) @ Q.mH
    return sqrt_rho


def is_state(rho):
    return rho.shape[-1] == 1


def fidelity(rho1, rho2):
    rho1_state = is_state(rho1)
    rho2_state = is_state(rho2)
    if rho1_state and rho2_state:
        f = rho1.mH @ rho2
        f *= f.conj()
    elif rho1_state:
        f = rho1.mH @ rho2 @ rho1
    elif rho2_state:
        f = rho2.mH @ rho1 @ rho2
    else:
        sq_rho1 = sqrtm(rho1)
        eigv = torch.linalg.eigvalsh(sq_rho1 @ rho2 @ sq_rho1)
        eigv[eigv < 0] = 0
        f = eigv.sqrt().sum(-1)

    f = cast_real(f)
    assert torch.all(f >= 0), f"Negative fidelity! {f.min()}"
    assert torch.all(f <= 1), f"Fidelity larger than 1! {f.max()}"





    return f


class IMLE:
    def __init__(self, hilbert_size, max_iterations):
        self.hilbert_size = hilbert_size
        self.max_iterations = max_iterations

    def sample(self, context_x, context_v, query_v):

        num_samples, num_context, v_size = context_v.shape

        context_v = context_v.view([num_samples, num_context, self.hilbert_size, self.hilbert_size, 2])
        context_v = context_v[..., 0] + 1j * context_v[..., 1]

        context_x = context_x.to(context_v)

        rho = [rand_dm(self.hilbert_size, 0.8) for _ in range(num_samples)]
        rho = to_torch(rho).to(context_v)

        for i in range(self.max_iterations):
            check_density(rho)
            guessed_val = expect(context_v, rho)

            ratio = context_x / guessed_val

            ratio = ratio.view([num_samples, num_context, 1, 1])


            R = (context_v * ratio).sum(1)
            rho = R @ rho @ R.mH
            rho = rho / trace(rho, cast=False).view([num_samples, 1, 1])



        return rho, None, None


def get_loss_fn(hilbert_size=32, imle_iterations=1000, return_val=False):
    iMLE = IMLE(hilbert_size, imle_iterations)

    def loss_fn(path, data):
        batch_size, num_samples, num_context = path.shape

        data = [d.to(torch.float64) for d in data[:3]] + data[3:]

        rho, _, _ = evaluate_backbone(path, data, iMLE)

        if return_val:
            context_v = data[0].view([batch_size * num_samples, num_context, hilbert_size, hilbert_size, 2])
            context_v = context_v[..., 0] + 1j * context_v[..., 1]
            val = expect(context_v, rho)
            return val

        rho = rho.view([batch_size, num_samples, hilbert_size, hilbert_size])
        rho_true = data[3].view([batch_size, 1, hilbert_size, 1]).to(rho)

        loss = 1 - fidelity(rho, rho_true).view([batch_size, num_samples])
        
        return loss

    return loss_fn
