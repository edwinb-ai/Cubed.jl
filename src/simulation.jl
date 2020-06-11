"""
    This is the main API that should be available to the end user.
It just receives the main information for the system to simulate, and the rest
is done internally.

It always computes and prints out the initial energy of the system.
"""
function simulate(
    syst::System,
    dynamic::Dynamic,
    potential::Potential;
    cycles::Integer = 200000
)
    # Assign device memory for virial and energy
    P = CuVector{Float32}(undef, syst.N)
    fill!(P, 0.0f0)
    syst.press = P
    E = CuVector{Float32}(undef, syst.N)
    fill!(E, 0.0f0)
    syst.ener = E

    # GPU variables
    nthreads = 512
    gpu_blocks = ceil(Int, syst.N / nthreads)

    # Create the positions array, first in host
    positions = Matrix{Float32}(undef, 3, syst.N)
    fill!(positions, 0.0f0)
    # Creat the forces array, in device
    forces = CuMatrix{Float32}(undef, 3, syst.N)
    fill!(forces, 0.0f0)
    # Initialize the positions as grid
    init!(positions, syst)
    # Send the positions array to device
    cu_positions = cu(positions)

    # ! Compute the initial energy of the system
    CUDA.@sync begin
        @cuda threads = nthreads blocks = gpu_blocks gpu_energy!(
            cu_positions,
            forces,
            syst.N,
            syst.L,
            syst.rc,
            potential.a,
            potential.b,
            potential.λ,
            potential.temp,
            syst.ener,
            syst.press,
        )
    end
    totale = sum(syst.ener) / syst.N
    println("Initial energy: $(totale)")

    # * Main loop
    move(
        cu_positions,
        forces,
        syst,
        dynamic, 
        potential, 
        cycles;
        filename = "$(syst.ρ)_$(syst.N).csv",
        thermal = 150000
    )
end
