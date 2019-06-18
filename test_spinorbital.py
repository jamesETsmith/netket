import netket as nk
import numpy as np
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import mcscf
from pyscf import fci
import pdb

L = 2

# Make integrals
atom = [["C", (0, 0, i)] for i in range(L)]
print(atom)
mol = gto.M(atom=atom, spin=L)
mf = scf.RHF(mol).run()

nmo = mf.mo_coeff.shape[0]
nelec = mol.nelec[0] + mol.nelec[1]
print(nmo, nelec)
eri = mol.intor("int2e_sph")

# h = np.einsum("pq,pi,qj", mf.get_hcore(), mf.mo_coeff, mf.mo_coeff)
g = ao2mo.incore.full(eri, mf.mo_coeff)  # .reshape(nmo ** 2, nmo ** 2)

h = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
# g = ao2mo.incore.kernel(mol, mf.mo_coeff)
cisolver = fci.direct_spin1.FCI(mol)
e_exact, ci = cisolver.kernel(h, g, h.shape[1], mol.nelec, ecore=mol.energy_nuc())
print("PySCF FCI Energy:", e_exact)

#
# NetKet
#

hi = nk.hilbert.SpinOrbital(
    graph=nk.graph.Hypercube(length=nmo * 2, n_dim=1), nelec=nelec
)

ham = nk.operator.QCHamiltonian(hilbert=hi, h=h, g=g.reshape(nmo ** 2, nmo ** 2))
# exit(0)

# Machine
ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Sampler
sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=ham)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.01)

# Variational Monte Carlo
gs = nk.variational.Vmc(
    hamiltonian=ham, sampler=sa, optimizer=op, n_samples=1000, diag_shift=0.01
)

# exit(0)
# pdb.set_trace()
gs.run(output_prefix="test", n_iter=300, save_params_every=10)
