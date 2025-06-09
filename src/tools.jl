using LinearAlgebra: normalize

using TensorOperations: @tensor
using TensorKit:
    ⊗,
    ←,
    DiagonalTensorMap,
    Sector,
    codomain,
    domain,
    fuse,
    id,
    isometry,
    permute,
    spacetype,
    truncbelow,
    truncdim
using MPSKit: correlation_length, expectation_value, leading_boundary
using MPSKitModels: S_exchange
using PEPSKit:
    CTMRGEnv,
    EAST,
    InfinitePEPS,
    InfiniteSquare,
    InfiniteWeightPEPS,
    LocalOperator,
    NORTH,
    SimpleUpdate,
    SOUTH,
    SUWeight,
    WEST,
    nearest_neighbours,
    next_nearest_neighbours,
    simpleupdate

function operator_with_ancilla(h; va0=nothing)
    vd = first(domain(h))
    va = isnothing(va0) ? vd' : va0
    vda = fuse(vd ⊗ va)
    h_pxa = h ⊗ id(va ⊗ va)
    u = isometry(vda, vd ⊗ va)
    @tensor h_pa[-1, -2; -3, -4] :=
        h_pxa[1, 2, 3, 4, 5, 6, 7, 8] *
        u[-1, 1, 3] *
        u[-2, 2, 4] *
        u'[5, 7, -3] *
        u'[6, 8, -4]
    return h_pa
end

function init_heisenberg(h_bond; J2=nothing)
    nrows, ncols = 2, 2
    h_pa = operator_with_ancilla(h_bond)
    vd = first(domain(h_bond))
    vD = oneunit(vd)

    id_vd = permute(id(vd), (1, 2), ())  # init with identity from physical to ancilla
    vda = fuse(codomain(id_vd))
    u = isometry(vda, codomain(id_vd))  # merge physical and ancilla legs
    virt = ones(domain(id_vd) ← vD ⊗ vD ⊗ vD' ⊗ vD')  # add 4 trivial virtual legs
    peps0_tensor = u * id_vd * virt

    physical_spaces = fill(vda, (nrows, ncols))
    lattice = InfiniteSquare(nrows, ncols)

    vertices0 = fill(peps0_tensor, (nrows, ncols))
    weights0 = SUWeight(
        map(Iterators.product(1:2, 1:nrows, 1:ncols)) do (dir, r, c)
            return DiagonalTensorMap(id(domain(vertices0[r, c])[dir]))
        end,
    )
    wpeps0 = InfiniteWeightPEPS(vertices0, weights0)

    terms = [CartesianIndex.(t) => h_pa for t in nearest_neighbours(lattice)]  # 1st order SU, 1st neighbor
    if !isnothing(J2)   # 1st order SU, 2nd neighbor
        append!(
            terms,
            [CartesianIndex.(t) => J2 * h_pa for t in next_nearest_neighbours(lattice)],
        )
    end
    su_hamilt = LocalOperator(physical_spaces, terms...)

    bond_operators = map(nearest_neighbours(lattice)) do inds
        return LocalOperator(physical_spaces, Tuple.(inds) => h_pa)
    end
    diag_operators = map(next_nearest_neighbours(lattice)) do inds
        return LocalOperator(physical_spaces, Tuple.(inds) => h_pa)
    end
    return wpeps0, su_hamilt, bond_operators, diag_operators
end

function init_env(ψ)
    v0 = oneunit(spacetype(first(ψ.A)))
    corners = fill(id(v0), (4, size(ψ)...))
    edges = map(Iterators.ProductIterator(([SOUTH, WEST, NORTH, EAST], ψ.A))) do (dir, t)
        vD = domain(t)[dir]
        return permute(id(vD ⊗ v0), (2, 3, 1), (4,))
    end
    return CTMRGEnv(corners, edges)
end

function converge_heisenberg(
    S::Type{<:Sector}, D::Integer, boundary_alg; β=1.0, dt=1e-3, su_cutoff=1e-10
)
    h = S_exchange(Float64, S; spin=1//2)
    wpeps0, su_hamilt, _, _ = init_heisenberg(h)
    su_alg = SimpleUpdate(
        dt, -Inf, Int(round(β / 2dt)), truncdim(D) & truncbelow(su_cutoff)
    )
    wpeps1, _ = simpleupdate(wpeps0, su_hamilt, su_alg; bipartite=true, check_interval=2^62)
    ψ = normalize(InfinitePEPS(wpeps1))
    env0 = init_env(ψ)
    @time env∞, _ = leading_boundary(env0, ψ; boundary_alg...)
    return wpeps1, env∞
end

function compute_obs(ψ, env, bond_operators, diag_operators; J1=1.0, J2=0.0)
    println("Computing first neighbor bond energies...")
    @time bond_energies = map(op -> expectation_value(ψ, op, env), bond_operators)

    println("Computing second neighbor bond energies...")
    @time diag_energies = map(op -> expectation_value(ψ, op, env), diag_operators)
    energy = (J1 * sum(bond_energies) + J2 * sum(diag_energies)) / 4

    println("Computing transfer spectrum...")
    @time ξ_h, ξ_v, λ_h, λ_v = correlation_length(ψ, env)
    ξ_h *= size(env, 2)
    ξ_v *= size(env, 3)
    return (;
        bond_energies=bond_energies,
        diag_energies=diag_energies,
        energy=energy,
        ξ_h=ξ_h,
        ξ_v=ξ_v,
        λ_h=real(λ_h),
        λ_v=real(λ_v),
    )
end
