import numpy as np

from qutip import coherent, coherent_dm, expect, Qobj, fidelity, rand_dm, displace, fock_dm
from qutip.wigner import wigner, qfunc

from tqdm.auto import tqdm

import matplotlib.pyplot as plt

def wigner_dm(hilbert_size, beta):
    D = displace(hilbert_size, beta)
    op = sum([((-1) ** i) * D * fock_dm(hilbert_size, i) * D.dag() for i in range(hilbert_size)])
    op = (op) * (2 / np.pi)
    return op


def cat(hilbert_size, alpha):
    psi = coherent(hilbert_size, alpha)
    psi += coherent(hilbert_size, -alpha)

    return psi.unit()


def binomial(hilbert_size):
    psi = Qobj(np.array([1/np.sqrt(2), 0, 0, 0, 1/np.sqrt(2)]))
    return psi.unit()


hilbert_size = 32
alpha = 2
psi = cat(hilbert_size, alpha)


grid = 32
xvec = np.linspace(-3, 3, grid)
yvec = np.linspace(-3, 3, grid)


X, Y = np.meshgrid(xvec, yvec)
betas = (X + 1j*Y).ravel()
m_ops = [coherent_dm(hilbert_size, beta) for beta in betas]
ops_numpy = [op.full() for op in m_ops]

data = expect(m_ops, psi)
rho_true = psi*psi.dag()


fidelities = []
max_iterations = 1000

rho = rand_dm(hilbert_size, 0.8)
fidelities.append(fidelity(rho_true, rho))
pbar = tqdm(range(max_iterations))


for i in range(max_iterations):
    guessed_val = expect(m_ops, rho)
    ratio = np.array(data) / np.array(guessed_val)

    R = Qobj(np.einsum("aij,a->ij", ops_numpy, ratio))
    rho = R * rho * R
    rho = rho / rho.tr()

    f = fidelity(rho, rho_true)
    fidelities.append(f)

    pbar.set_description("Fidelity iMLE {:.4f}".format(f))
    pbar.update()


iterations = np.arange(len(fidelities))
plt.plot(iterations, fidelities, color="blue", label="iMLE")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Fidelity")
plt.ylim(0, 1.02)
plt.grid(which='minor', alpha=0.2)
plt.grid(which='major', alpha=0.2)
plt.xscale('log')
plt.savefig("imle.png")
plt.close()


titles = ["original", "reconstructed"]
data = [rho_true, rho]
data = np.stack([qfunc(r, xvec, yvec, g=2) for r in data])

vmin, vmax = data.min(), data.max()

data = list(zip(titles, data))


fig, axs = plt.subplots(1, len(data), sharey=True)
fig.suptitle("Q function")
fig.tight_layout()

for ax, (title, q) in zip(axs, data):
    im = ax.pcolor(xvec, yvec, q, vmin=vmin, vmax=vmax, cmap="jet", shading="auto")
    ax.set_title(title)

fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.1)
fig.savefig("q.png")
