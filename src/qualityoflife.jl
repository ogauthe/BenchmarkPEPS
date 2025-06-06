using LinearAlgebra: diag
using Printf: Printf

using TensorKit: block, blocksectors, codomain, dim, domain, numin, numout, sectortype

function pprint_vec(vec::AbstractVector; digits=8)
    fmt = Printf.Format("% ." * string(digits) * "f")
    formatted = [Printf.format(fmt, v) for v in vec]
    return "[" * join(formatted, "  ") * "]"
end

function pprint_tensormap(io, t)
    print(io, nameof(typeof(t)))
    print(io, "{$(eltype(t)), $(sectortype(t)), $(numout(t)), $(numin(t))}")
    return print(io, " shape ", dim.(Tuple(codomain(t))), dim.(Tuple(domain(t))))
end

function pprint_weights(wpeps)
    println("Weights:")
    map(CartesianIndices((2, 2, 2))) do inds
        println(inds, ":")
        w = wpeps.weights[inds]
        wn = sum(sum(block(w, s)) * dim(s) for s in blocksectors(w))
        map(blocksectors(w)) do s
            println("$s => ", pprint_vec(diag(block(w/wn, s))))
        end
    end
    return nothing
end

function pprint_corners(env)
    cspaces = unique(
        only(domain(env.corners[inds])) for inds in CartesianIndices((2, 2, 2))
    )
    println("chi values: ", dim.(cspaces))
    println("Corner spaces:")
    println.(cspaces)
    return nothing
end
