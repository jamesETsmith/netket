from time import time
import netket as nk
import numpy as np
from functools import reduce
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import mcscf
from pyscf import fci
from pyscf.shciscf.shci import SHCI

L = 6
Atom = "H"
file_base = "{}{}".format(Atom, L)
spin = 0

# Make integrals
np.random.seed(20)
# atom = "C 0 0 0; C 0 0 1.25"
atom = [[Atom, (0, 0, i)] for i in range(L)]
# print(atom)
mol = gto.M(
    atom=atom,
    spin=spin,
    symmetry=False,
    charge=0,
    verbose=5,
    output="{}/_pyscf.out".format(file_base),
)
mf = scf.RHF(mol).run()

nmo = mf.mo_coeff.shape[0]
nelec = mol.nelec[0] + mol.nelec[1]
# print(nmo, nelec)
eri = mol.intor("int2e_sph")
g = ao2mo.full(eri, mf.mo_coeff, verbose=0, compact=False)

t = mol.intor_symmetric("int1e_kin")
v = mol.intor_symmetric("int1e_nuc")
h = reduce(np.dot, (mf.mo_coeff.T, t + v, mf.mo_coeff))

if False:
    cisolver = fci.direct_spin1.FCI(mol)
    e_exact, ci = cisolver.kernel(h, g, h.shape[1], mol.nelec, ecore=mol.energy_nuc())
    rdm1, rdm2 = cisolver.make_rdm12(ci, h.shape[1], mol.nelec)

    print("PySCF HF Electronic Energy", mf.e_tot - mol.energy_nuc())
    print("PySCF FCI Total Energy:", e_exact)
    print("PySCF FCI Electronic Energy:", e_exact - mol.energy_nuc())
    print("Nuclear Repulsion =", mol.energy_nuc())
    print("<T + V_ne>", np.einsum("ij,ji", h, rdm1))
    print("<V_ee>", np.einsum("pqrs,pqrs", g, rdm2))
    print("Number of Dets in FCI expansions=", ci.shape)
    exit(0)

if False:
    mc = mcscf.CASCI(mf, mf.mo_coeff.shape[0], mol.nelectron)
    mc.fcisolver = SHCI(mol)
    mc.fcisolver.sweep_iter = [0]
    mc.fcisolver.sweep_epsilon = [0]
    mc.kernel()


#
# NetKet
#

hi = nk.hilbert.SpinOrbital(
    graph=nk.graph.Hypercube(length=nmo * 2, n_dim=1), nelec=nelec, sz=spin
)

ham = nk.operator.QCHamiltonian(
    hilbert=hi,
    h=h,
    g=g.reshape(nmo ** 2, nmo ** 2),
    sz=spin,
    e0=mf.energy_nuc(),
    nelec=mol.nelectron,
)
# exit(0)

# Machine
ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
ma.init_random_parameters(seed=12345, sigma=0.01)

# np_ma1 = np.sort(np.power(np.abs(ma.to_array()), 2))
# ma.load("H6/H6.wf")
# np_ma2 = np.sort(np.power(np.abs(ma.to_array()), 2))


# def test_conf(conf):
#     sz = 0
#     nelec = 0
#     for i, c in enumerate(conf):
#         if c == 1:
#             nelec += 1
#             if i % 2 == 0:
#                 sz += 1
#             elif i % 2 == 1:
#                 sz -= 1

#     if sz != 0 or nelec != 6:
#         return False
#     else:
#         return True


# res = nk.exact.full_ed(ham, first_n=4, compute_eigenvectors=True)
# print(res.eigenvalues)
# print(res.eigenvectors[0])
# gs_raw = res.eigenvectors[0]
# gs = np.abs(
#     np.array(
#         [el for el, state in zip(res.eigenvectors[0], hi.states()) if test_conf(state)]
#     )
# )
# gs /= gs.max()

# states = [state for state in hi.states() if test_conf(state)]
# # print(list(hi.states())[0])
# # print(len(states))

# logvals = np.array([ma.log_val(state) for state in states])
# np_ma3 = np.sort(np.abs(np.exp(logvals - logvals.max())) ** 2)
# np_ma3_arg = np.argsort(np.abs(np.exp(logvals - logvals.max())) ** 2)

# print(np_ma3_arg[-1])
# print(logvals[np_ma3_arg[-1]])
# print(states[np_ma3_arg[-1]])
# print(logvals[np_ma3_arg[-2]])


# import matplotlib.pyplot as plt

# plt.figure()
# plt.plot(np_ma3, "o-")
# plt.plot(gs ** 2, "o-")
# # plt.ylim([-1e-4, 1e-4])
# plt.savefig("wf_test.png")
# plt.xlim([380, 400])

# # print(states[-20:])
# # print(np.abs(logvals[-20:]))
# exit(0)

# Sampler
sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=ham)
# sa = nk.sampler.MetropolisHamiltonianPt(machine=ma, hamiltonian=ham, n_replicas=10)

# Optimizer
# op = nk.optimizer.Sgd(learning_rate=0.1)
# op = nk.optimizer.Momentum(learning_rate=0.01)
op = nk.optimizer.AmsGrad(learning_rate=0.01, beta1=0.95, beta2=0.99, epscut=1e-8)

# Variational Monte Carlo
# pars = [ n_samples, n_iter, save_params_every ]
# pars = [20000, 1000, 100]  # Real run
pars = [10000, 100, 10]  # Small run
# pars = [1000, 10, 10]  # Profiling
# pars = [1, 1, 1]  # For testing ED

gs = nk.variational.Vmc(
    hamiltonian=ham,
    sampler=sa,
    optimizer=op,
    n_samples=pars[0],
    diag_shift=0.01,
    method="Sr",
)

t0 = time()
gs.run(
    output_prefix="{}/{}".format(file_base, file_base),
    n_iter=pars[1],
    save_params_every=pars[2],
)

print("Time fo NetKet", time() - t0)

