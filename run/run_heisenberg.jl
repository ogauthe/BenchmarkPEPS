using LinearAlgebra: LinearAlgebra, diag, normalize
using Pkg: Pkg
using Printf: @sprintf
using Dates: Dates
using Logging: Logging
using MKL

using JSON: JSON
using TensorKit:
    ←,
    ⊗,
    Trivial,
    SU2Irrep,
    U1Irrep,
    TensorMap,
    Vect,
    block,
    blocksectors,
    dim,
    domain,
    eigh,
    truncbelow,
    truncdim
using MPSKit: leading_boundary
using MPSKitModels: S_exchange
using PEPSKit: InfinitePEPS, SimpleUpdate, simpleupdate

using BenchmarkPEPS:
    compute_obs,
    init_env,
    init_sim_1stnei,
    pprint_corners,
    pprint_diag,
    pprint_vec,
    pprint_weights

# ======================================================================================
# log

println("\n# ", "="^90)
println(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
println("Julia $VERSION")
@show Base.julia_cmd()
@show Threads.nthreads()
@show LinearAlgebra.BLAS.get_num_threads()
Pkg.status()
flush(stdout)

# ======================================================================================
# Parameters

if length(ARGS) < 1
    println("Missing input file: use default")
    json_input = String(dirname(@__FILE__)) * "/../input_sample/input_sample_heisenberg.jl"
else
    json_input = String(ARGS[1])
end

println("\nLoad input parameters from file $json_input")

input_data = JSON.parsefile(json_input)
map(k -> println(k, ": ", input_data[k]), sort(collect(keys(input_data))))

model = "Heisenberg"
J1 = 1.0
ctm_tol::Float64 = Float64(input_data["ctm_tol"])
beta_values::Vector{Float64} = Vector{Float64}(input_data["beta_values"])
ctm_cutoff::Float64 = Float64(input_data["ctm_cutoff"])
dt::Float64 = Float64(input_data["dt"])
Dmax::Int = Int(input_data["Dmax"])
su_cutoff::Float64 = Float64(input_data["su_cutoff"])
χ::Int = Int(input_data["chi"])
symmetry::String = String(input_data["symmetry"])
ctm_maxiter::Int = Int(input_data["ctm_maxiter"])
rad::String = String(input_data["rad"])

# ======================================================================================
# misc

function pprint_observables(res::NamedTuple)
    println("observables:")
    println("energy = ", res.energy)
    println("bond_energies: ", pprint_vec(res.bond_energies))
    println("diag_energies: ", pprint_vec(res.diag_energies))
    println("ξ_h: ", pprint_vec(res.ξ_h))
    return println("ξ_v: ", pprint_vec(res.ξ_v))
end

function get_savefile(output_data)
    D = string(output_data["Dmax"])
    β = @sprintf("%.4f", output_data["beta"])
    χ = string(output_data["chi"])
    return rad * "_D$D" * "_beta$β" * "_chi$χ" * ".json"
end

# ======================================================================================
# set metadata

metadata = Dict([
    "Dmax" => Dmax,
    "J1" => J1,
    "chi" => χ,
    "ctm_cutoff" => ctm_cutoff,
    "ctm_tol" => ctm_tol,
    "dt" => dt,
    "model" => model,
    "su_cutoff" => su_cutoff,
    "symmetry" => symmetry,
])
println("\nmetadata:")
display(metadata)
@show beta_values

# ======================================================================================
# Init SU and CTM

println("\n# ", "="^90)
println("Initializing SimpleUpdate...")
flush(stdout)

@time begin
    su_trscheme = truncdim(Dmax) & truncbelow(su_cutoff)
    ctm_trscheme = truncdim(χ) & truncbelow(ctm_cutoff)
    boundary_alg = (; tol=ctm_tol, trscheme=ctm_trscheme, maxiter=ctm_maxiter)
    lastβ = 0.0
    env = nothing
    if symmetry == "SU2"
        hsector = SU2Irrep
    elseif symmetry == "U1"
        hsector = U1Irrep
    elseif symmetry == "Trivial"
        hsector = Trivial
    end
    h = S_exchange(Float64, hsector; spin=1//2)
    spec, _ = eigh(h)
    println("Bond Hamiltonian spectrum:")
    pprint_diag(spec)
    wpeps, su_hamilt, bond_operators, diag_operators = init_sim_1stnei(h)
    nrows, ncols = size(wpeps)
end
flush(stdout)

# ======================================================================================
# run SU

for β in beta_values
    println("\n# ", "="^92, "\nbeta = $β")
    Δβ = β - lastβ
    su_niter = Int(round(Δβ / 2dt; digits=10))

    # ===============  SimpleUpdate  =======================================================
    println("\nRunning SimpleUpdate from β=$lastβ to β=$β...")
    flush(stdout)

    su_alg = SimpleUpdate(dt, -Inf, su_niter, su_trscheme)
    Logging.with_logger(Logging.ConsoleLogger(stdout, Logging.Error)) do
        @time global wpeps, _ = simpleupdate(
            wpeps, su_hamilt, su_alg; bipartite=true, check_interval=2^62
        )
    end
    pprint_weights(wpeps)

    # =====================  CTMRG  ========================================================
    println("\nRunning CTMRG...")
    flush(stdout)

    ψ = normalize(InfinitePEPS(wpeps)) # absorb the weights
    global env = isnothing(env) ? init_env(ψ) : env
    Logging.with_logger(Logging.ConsoleLogger(stdout, Logging.Info)) do
        @time global env, info_ctmrg = leading_boundary(env, ψ; boundary_alg...)
    end
    pprint_corners(env)

    # ====================  observables   ==================================================
    println("\nComputing observables...")
    flush(stdout)

    observables = compute_obs(ψ, env, bond_operators, diag_operators)
    println()
    pprint_observables(observables)

    # ======================  save data  ===================================================
    metadata["beta"] = β
    output_data = merge(metadata, Dict(string(k) => v for (k, v) in pairs(observables)))
    savefile = get_savefile(output_data)
    open(savefile, "w") do io
        JSON.print(io, output_data, 4)
    end
    println("\nResults saved in $savefile")
    flush(stdout)

    global lastβ = β
end

println("\n# ", "="^40, "  END  ", "="^41)
println(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
