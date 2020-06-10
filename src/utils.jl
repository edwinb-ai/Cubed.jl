function init!(positions::AbstractArray, syst::System)
    dist = cbrt(1.0f0 / syst.ρ)
    dist_half = dist / 2.0f0

    # Define the first positions
    positions[1, 1] = -syst.rc + dist_half
    positions[2, 1] = -syst.rc + dist_half
    positions[3, 1] = -syst.rc + dist_half

    # Create a complete lattice
    @inbounds for i = 2:(syst.N - 1)
        positions[1, i] = positions[1, i - 1] + dist
        positions[2, i] = positions[2, i - 1]
        positions[3, i] = positions[3, i - 1]

        if positions[1, i] > syst.rc
            positions[1, i] = positions[1, 1]
            positions[2, i] = positions[2, i - 1] + dist

            if positions[2, i] > syst.rc
                positions[1, i] = positions[1, 1]
                positions[2, i] = positions[2, 1]
                positions[3, i] = positions[3, i - 1] + dist
            end
        end
    end

    return nothing
end

function move(
    positions::AbstractArray,
    forces::AbstractArray,
    syst::System, 
    dynamics::Dynamic,
    potential::Potential,
    cycles::Integer;
    filename::String = nothing,
    thermal::Integer = Int(cycles / 2)
)
    # * Accumulation variables
    total_energy = 0.0f0
    total_pressure = 0.0f0
    big_z = 0.0f0
    total_z = 0.0f0
    total_virial = 0.0f0
    samples = 0

    # GPU variables for the kernels
    nthreads = 512
    gpu_blocks = ceil(Int, syst.N / nthreads)

    # Allocate memory for random numbers
    rnd_matrix = Matrix{Float32}(undef, 3, syst.N)

    # ! Main loop
    @showprogress for i = 1:cycles
        # * Create array of random numbers
        Random.randn!(syst.rng, rnd_matrix)
        rnd_matrix .*= sqrt(2.0f0 * dynamics.τ)
        cu_rand_matrix = cu(rnd_matrix)

        # * Move the particles following the Ermak-McCammon algorithm
        CUDA.@sync begin
            @cuda threads = nthreads blocks = gpu_blocks gpu_ermak!(
                positions,
                forces,
                syst.N,
                syst.L,
                dynamics.τ,
                cu_rand_matrix,
            )
        end
        # ! Always set forces to zero
        fill!(forces, 0.0f0)
        
        # ! Compute the energy and forces, O(N^2) complexity
        CUDA.@sync begin
            @cuda threads = nthreads blocks = gpu_blocks gpu_energy!(
                positions,
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
        # * Save to file
        if i >= thermal
            samples += 1

            # * Update the total energy of the system
            total_energy = sum(syst.ener) / 2.0f0

            # * Extract the virial from the kernel
            total_virial = sum(syst.press)
            total_virial /= (3.0f0 * syst.N)
            big_z = 1.0f0 + total_virial
            total_z += big_z # This is just an accumulator
            # * Update the total pressure of the system
            total_pressure = big_z * syst.ρ

            if !isnothing(filename)
                open(filename, "a") do io
                    filener = total_energy / (syst.N * samples)
                    println(io, "$(filener),$(total_pressure),$(big_z)")
                end
            end
        end
    end

    # ! Adjust results as averages
    total_energy /= syst.N
    total_z /= samples

    # * Show results
    println("Energy: $(total_energy)")
    println("Pressure: $(total_pressure)")
    println("Compressibility: $(big_z)")
    println("Compressibility (average): $(total_z)")
end
