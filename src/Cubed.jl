module Cubed

using Random
using RandomNumbers.Xorshifts
using ProgressMeter
using CUDA

include("types.jl")
export System, Dynamic, Potential
include("kernels.jl")
export gpu_energy!, gpu_ermak!
include("utils.jl")
export init!, move

function run()
    CUDA.allowscalar(false)

    # * Simulation parameters
    N = 1728 # number of particles
    ϕ = 0.4f0 # filling/packing fraction
    ρ = ϕ * 6.0f0 / π # reduced density
    volumen = N / ρ # reduced volume
    Lbox = ∛volumen # Cubic box length
    rc = Lbox / 2.0f0
    P = CuVector{Float32}(undef, N)
    CUDA.fill!(P, 0.0f0)
    E = CuVector{Float32}(undef, N)
    CUDA.fill!(E, 0.0f0)

    # gpu_blocks = 1
    nthreads = 256
    gpu_blocks = ceil(Int, N / nthreads)

    # Create a rng
    rng = Xorshift1024Star()
    # Initialize the system object
    syst = System(N, Lbox, ρ, volumen, P, rc, rng, E)
    dynamic = Dynamic(0.000005f0)
    λ = 50
    b = convert(Float32, λ / (λ - 1.0f0))
    a = convert(Float32, λ * ^(b, λ - 1.0f0))
    potential = Potential(a, b, 1.4737f0, 50)

    # Create the positions array, first in host
    positions = Matrix{Float32}(undef, 3, N)
    fill!(positions, 0.0f0)
    # Creat the forces array, in device
    forces = CuMatrix{Float32}(undef, 3, N)
    CUDA.fill!(forces, 0.0f0)
    # Initialize the positions as grid
    init!(positions, syst)
    # Send the positions array to device
    cu_positions = CuArray(positions)

    # ! Compute the initial energy of the system
    CUDA.@sync begin
        @cuda threads=nthreads blocks=gpu_blocks gpu_energy!(cu_positions,
            forces,
            syst.N,
            syst.L,
            syst.rc,
            potential.a,
            potential.b,
            potential.λ,
            potential.temp,
            E,
            P,
        )
    end
    totale = sum(E) / syst.N
    println("Initial energy: $(totale)")

    # * Main loop
    move(
        positions,
        forces,
        syst,
        dynamic, 
        potential, 
        200000;
        filename = "$(ρ)_$(N).csv",
        thermal = 150000,
    )
end

run()

end # module
