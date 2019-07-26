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
import pdb

L = 4

# Make integrals
np.random.seed(20)
# atom = [["H", (np.random.random(1), np.random.random(1), i)] for i in range(L)]
atom = [["H", (0, 0, i)] for i in range(L)]
print(atom)
mol = gto.M(
    atom=atom, spin=0, symmetry=False, charge=0, verbose=5, output="spinorb.out"
)
mf = scf.RHF(mol).run()

nmo = mf.mo_coeff.shape[0]
nelec = mol.nelec[0] + mol.nelec[1]
print(nmo, nelec)
eri = mol.intor("int2e_sph")
g = ao2mo.full(eri, mf.mo_coeff, verbose=0, compact=False)

t = mol.intor_symmetric("int1e_kin")
v = mol.intor_symmetric("int1e_nuc")
h = reduce(np.dot, (mf.mo_coeff.T, t + v, mf.mo_coeff))

if True:
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
    # exit(0)

# exit(0)
if True:
    mc = mcscf.CASCI(mf, mf.mo_coeff.shape[0], mol.nelectron)
    mc.fcisolver = SHCI(mol)
    mc.fcisolver.sweep_iter = [0]
    mc.fcisolver.sweep_epsilon = [0]
    mc.kernel()


# g *= 0.5

#
# NetKet
#

hi = nk.hilbert.SpinOrbital(
    graph=nk.graph.Hypercube(length=nmo * 2, n_dim=1), nelec=nelec
)

ham = nk.operator.QCHamiltonian(
    hilbert=hi,
    h=h,
    g=g.reshape(nmo ** 2, nmo ** 2),
    e0=mf.energy_nuc(),
    nelec=mol.nelectron,
)
# exit(0)

# Machine
ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Sampler
sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=ham)

# Optimizer
# op = nk.optimizer.Sgd(learning_rate=0.01)
op = nk.optimizer.AmsGrad(learning_rate=0.01, beta1=0.8, beta2=0.95, epscut=1e-8)

# Variational Monte Carlo
gs = nk.variational.Vmc(
    hamiltonian=ham, sampler=sa, optimizer=op, n_samples=1, diag_shift=0.01
)

# ob = nk.exact.full_ed(operator=ham)
# ob = nk.exact.lanczos_ed(operator=ham)
# print(ob.eigenvalues)
# exit(0)
# pdb.set_trace()

t0 = time()
gs.run(output_prefix="test", n_iter=1, save_params_every=20)
print("Time fo NetKet", time() - t0)

