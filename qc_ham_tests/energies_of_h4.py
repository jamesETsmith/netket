from time import time
import netket as nk
import numpy as np
from functools import reduce
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import mcscf
from pyscf import fci


def get_energy(spin):
    """
    """
    L = 4

    # Make integrals
    atom = [["H", (0, 0, i)] for i in range(L)]
    print(atom)
    mol = gto.M(
        atom=atom, spin=spin, symmetry=False, charge=0, verbose=5, output="spinorb.out"
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

    cisolver = fci.direct_spin1.FCI(mol)
    cisolver.threads = 4
    e_exact, ci = cisolver.kernel(h, g, h.shape[1], mol.nelec, ecore=mol.energy_nuc())
    rdm1, rdm2 = cisolver.make_rdm12(ci, h.shape[1], mol.nelec)

    print("PySCF HF Electronic Energy", mf.e_tot - mol.energy_nuc())
    print("PySCF FCI Total Energy:", e_exact)
    print("PySCF FCI Electronic Energy:", e_exact - mol.energy_nuc())
    print("Nuclear Repulsion =", mol.energy_nuc())
    print("<T + V_ne>", np.einsum("ij,ji", h, rdm1))
    print("<V_ee>", np.einsum("pqrs,pqrs", g, rdm2))
    print("Number of Dets in FCI expansions=", ci.shape)

    return e_exact


if __name__ == "__main__":
    energies = [get_energy(s) for s in [0, 2, 4]]
    print(energies)