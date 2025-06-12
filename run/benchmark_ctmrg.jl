using LinearAlgebra: normalize
using Test: @test

using TensorKit:
    AbstractTensorMap,
    AdjointTensorMap,
    DiagonalTensorMap,
    SU2Irrep,
    TensorMap,
    codomain,
    dim,
    domain,
    permute,
    truncdim,
    truncbelow
using BenchmarkTools: @benchmark
using MPSKit: leading_boundary
using PEPSKit: PEPSKit, EnlargedCorner, InfiniteSquareNetwork, InfinitePEPS

using BenchmarkPEPS:
    contract_enlarged_corner,
    converge_heisenberg,
    pprint_tensormap,
    pprint_corners,
    pprint_weights

# Quality of life type piracy
Base.size(t::AbstractTensorMap) = dim.(Tuple(codomain(t))), dim.(Tuple(domain(t)))

Base.show(io::IO, t::AbstractTensorMap) = pprint_tensormap(io, t)
Base.show(io::IO, t::TensorMap) = pprint_tensormap(io, t)
Base.show(io::IO, t::AdjointTensorMap) = pprint_tensormap(io, t)
Base.show(io::IO, t::DiagonalTensorMap) = pprint_tensormap(io, t)

function Base.show(io::IO, ::MIME"text/plain", t::AbstractTensorMap)
    show(io, t)
    println(io, "\ncodomain = ", codomain(t))
    return print(io, "domain = ", domain(t))
end

sector = SU2Irrep
D = 4
boundary_alg = (; tol=1e-7, trscheme=truncdim(D^2) & truncbelow(1e-12), maxiter=5);

wpeps, env = converge_heisenberg(sector, D, boundary_alg)
pprint_weights(wpeps)
pprint_corners(env)

network = InfiniteSquareNetwork(normalize(InfinitePEPS(wpeps)))
ctm_alg = PEPSKit.select_algorithm(leading_boundary, env; boundary_alg...);

# precompute enlarged corners and projectors
enlarged_corners = PEPSKit.dtmap(PEPSKit.eachcoordinate(network, 1:4)) do idx
    return TensorMap(PEPSKit.EnlargedCorner(network, env, idx), idx[1])
end;
projectors, info = PEPSKit.simultaneous_projectors(
    enlarged_corners, env, ctm_alg.projector_alg
)

Q = EnlargedCorner(network, env, (1, 1, 1))
ec1 = TensorMap(Q, 1)
ec2 = contract_enlarged_corner(Q)
@test ec1 ≈ ec2

println("benchmark: contract 1 enlarged corner")
@benchmark ec1 = TensorMap(Q, 1)
@benchmark ec2 = contract_enlarged_corner(Q)

println("benchmark: contract 1 half environment (=1 matmul)")
ec = (enlarged_corners[1, 1, 1], enlarged_corners[2, 1, 2])
@benchmark PEPSKit.half_infinite_environment(ec...)

println("benchmark: construct 1 pair of projectors (includes 1 matmul & 1 full SVD)")
@benchmark PEPSKit.compute_projector(ec, (1, 1, 1), ctm_alg.projector_alg)

println("benchmark: renormalize 1 edge")
@benchmark PEPSKit.renormalize_north_edge((1, 1), env, projectors..., network)

println("benchmark: permute 1 enlarged corner (size chi^2*D^4)")
perm = (2, 4, 3), (5, 1, 6)
@benchmark permute(ec1, perm)
