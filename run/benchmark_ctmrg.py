import numpy as np

from frostspin import AsymmetricTensor, SU2_SymmetricTensor, U1_SymmetricTensor
from frostspin.simple_update.simple_update import SimpleUpdate
from frostspin.ctmrg.ctmrg import CTMRG
from frostspin.ctmrg.ctm_renormalize import construct_projectors, renormalize_T1


sds22 = np.array(
    [
        [0.25, 0.0, 0.0, 0.0],
        [0.0, -0.25, 0.5, 0.0],
        [0.0, 0.5, -0.25, 0.0],
        [0.0, 0.0, 0.0, 0.25],
    ]
)
sds22t = sds22.reshape(2, 2, 2, 2)


def converge_env(
    sector,
    D,
    chi,
    beta=1.0,
    dt=1e-3,
    su_cutoff=1e-10,
    ctm_cutoff=1e-12,
    ctm_niter=10,
):
    if sector == "SU2":
        r2 = np.array([[1], [2]])
        h = SU2_SymmetricTensor.from_array(sds22t, (r2, r2), (r2, r2))
    elif sector == "U1":
        r2 = np.array([1, -1], dtype=np.int8)
        h = U1_SymmetricTensor.from_array(sds22t, (r2, r2), (r2, r2))
    elif sector == "Trivial":
        r2 = np.array([2])
        h = AsymmetricTensor.from_array(sds22t, (r2, r2), (r2, r2))

    su = SimpleUpdate.square_lattice_first_neighbor(
        h, D, dt, rcutoff=su_cutoff, degen_ratio=0.999
    )
    su.evolve(beta)

    ctm = CTMRG.from_elementary_tensors(
        "AB\nBA",
        su.get_tensors(),
        chi,
        block_chi_ratio=1.1,
        block_ncv_ratio=2.0,
        cutoff=ctm_cutoff,
        degen_ratio=0.999,
    )
    for i in range(ctm_niter):
        ctm.iterate()
    return su, ctm


sector = "SU2"
D = 4
chi = D**2


su, ctm = converge_env(sector, D, chi)
dr = ctm.construct_enlarged_dr(0, 0, free_memory=True)
ur = ctm.construct_enlarged_ur(0, 0, free_memory=True)
ul = ctm.construct_enlarged_ul(0, 0, free_memory=True)
dl = ctm.construct_enlarged_dl(0, 0, free_memory=True)
P0, Pt0 = construct_projectors(
    dr,
    ur,
    ul,
    dl,
    chi,
    ctm.block_chi_ratio,
    ctm.ncv_ratio,
    ctm.cutoff,
    ctm.degen_ratio,
    ctm.get_C2(2, 1),
)
P1, Pt1 = construct_projectors(
    ctm.construct_enlarged_dr(1, 0, free_memory=True),
    ctm.construct_enlarged_ur(1, 0, free_memory=True),
    ctm.construct_enlarged_ul(1, 0, free_memory=True),
    ctm.construct_enlarged_dl(1, 0, free_memory=True),
    chi,
    ctm.block_chi_ratio,
    ctm.ncv_ratio,
    ctm.cutoff,
    ctm.degen_ratio,
    ctm.get_C2(3, 1),
)


# magic command %timeit can only be used in IPython
print("benchmark: contract 1 enlarged corner")
# %timeit ctm.construct_enlarged_dr(0,0, free_memory=True)

print("benchmark: contract 1 half environment (=1 matmul)")
# %timeit halfr = dr @ ur

print("benchmark: construct 1 pair of projectors (includes 3 matmul & 1 partial SVD)")
# %timeit P, Pt = construct_projectors(dr, ur, ul, dl, chi, ctm.block_chi_ratio, ctm.ncv_ratio, ctm.cutoff, ctm.degen_ratio, ctm.get_C2(2, 1))

print("benchmark: renormalize 1 edge")
# %timeit renormalize_T1(Pt0, ctm.get_T1(-2, 0), ctm.get_A(-2,1), P1)

print("benchmark: permute 1 enlarged corner (size chi^2*D^4)")
# %timeit ul.permute((1, 3, 2), (4, 0, 5))
