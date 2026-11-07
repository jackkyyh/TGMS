import torch
from adaptive.dataset import DatasetAdapter

from qutip import coherent, coherent_dm, displace, fock_dm

from .utils import check_density, expect, to_torch



class Dataset(DatasetAdapter):
    def __init__(self, num_states=100, hilbert_size=32, xrange=(-3, 3), yrange=(-3, 3), grid=32):
        super().__init__(
            num_states=num_states,
            num_context=grid**2,
            x_size=1,
            v_size=2 * hilbert_size**2,
            q_v_size=0,
            q_x_size=hilbert_size,
            num_queries=1,
        )

        xvec = torch.linspace(*xrange, grid)
        yvec = torch.linspace(*yrange, grid)
        X, Y = torch.meshgrid(xvec, yvec, indexing="ij")

        alpha = torch.rand(num_states) * 1
        alpha = alpha * torch.exp(1j * torch.rand_like(alpha) * 2 * torch.pi)
        alpha += -1.5

        psi = cat_code(hilbert_size, alpha)
        psi = to_torch(psi)

        betas = (X + 1j * Y).T.ravel()
        m_ops = to_torch([coherent_dm(hilbert_size, beta) for beta in betas])


        exp = expect(m_ops.unsqueeze(0), psi)
        self.x = exp.view([num_states, self.num_context, self.x_size]).float()

        m_ops = torch.stack([m_ops.real, m_ops.imag], dim=-1)
        m_ops = m_ops.view([self.num_context, -1])

        self.v = m_ops
        self.v = self.v.view([self.num_context, self.v_size]).float()

        self.query_x = psi.view([num_states, self.num_queries, self.q_x_size])

        self.phase = alpha

    def __getitem__(self, idx):
        x = self.x[idx]
        v = self.v.expand([*x.shape[:-1], self.v_size])
        q_x = self.query_x[idx]
        q_v = torch.zeros([*v.shape[:-2], self.num_queries, self.q_v_size], device=q_x.device)

        aux = {k: getter(idx) for k, getter in self.aux_getter.items()}
        return (v, x, q_v, q_x) + ((aux,) if aux else ())

    def to(self, device):
        self.v = self.v.to(device)
        self.x = self.x.to(device)
        self.query_x = self.query_x.to(device)
        return self
    
    
    def get_phase(self, enabled):
        if enabled:
            self.aux_getter["phase"] = lambda idx: self.phase[idx]
        else:
            self.aux_getter.pop("phase", None)



def cat_code(hilbert_size, alpha):
    try:
        word = [cat_code(hilbert_size, a) for a in alpha]
    except:
        word = sum([
            coherent(hilbert_size, alpha),
            coherent(hilbert_size, -alpha),
            coherent(hilbert_size, 1j * alpha),
            coherent(hilbert_size, -1j * alpha),
        ]).unit()
    return word