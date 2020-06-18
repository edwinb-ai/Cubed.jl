"""
This constitutes the MAIN LOOP for all the simulation, it creates storage variables
as well as some arrays to compute the full evolution of the system. Most of the operations here
are carried out on the GPU device, except for the accumulation operations for the compressibility
factor and other things.
"""
function move!(
    positions::AbstractArray,
    forces::AbstractArray,
    syst::System, 
    dynamics::Dynamic,
    potential::Potential,
    cycles::Integer;
    filenames::AbstractArray = nothing,
    thermal::Integer = Int(cycles / 2),
    msd::Bool = false,
    logging::Bool = false,
)
# * Accumulation variables
    total_energy = 0.0f0
    total_pressure = 0.0f0
    big_z = 0.0f0
    total_z = 0.0f0
    total_virial = 0.0f0
    samples = 0
# MSD
    msdval = 0.0f0
    ref_pos = Matrix{Float32}(undef, 3, syst.N)
    time_acc = 0.0f0
    hst_positions = Matrix{Float32}(undef, 3, syst.N)

# GPU variables for the kernels
    nthreads = 512
    gpu_blocks = ceil(Int, syst.N / nthreads)

# Allocate memory for random numbers
    rnd_matrix = Matrix{Float32}(undef, 3, syst.N)

# When equilibrating, always use periodic boundary conditions
    pbc = true

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
            pbc
        )
        end
    # ! Always set forces to zero
        CUDA.fill!(forces, 0.0f0)
    
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

    # * Start accumulating averages
        if i >= thermal
        # ! Turn off periodic boundary conditions
            pbc = false
        
        # ! Update the samples counter
            samples += 1

        # * Update the total energy of the system
            total_energy = sum(syst.ener)

        # * Extract the virial from the kernel
            total_virial = sum(syst.press)
            total_virial /= (3.0f0 * syst.N)
            big_z = 1.0f0 + total_virial
            total_z += big_z # This is just an accumulator
        # * Update the total pressure of the system
            total_pressure = big_z * syst.ρ

        # ! Save energy, pressure and compressibility to file
            open(filenames[1], "a") do io
                filener = total_energy / (syst.N * samples)
                println(io, "$(filener),$(total_pressure),$(big_z)")
            end

            if msd
            # * Reference positions for the MSD, at first time step
                if samples == 1
                    ref_pos = Array(positions)
                end

            # * Accumulate MSD
                if samples > 1
                    time_acc += dynamics.τ
                    hst_positions = Array(positions)
                    msdval = msd!(hst_positions, ref_pos)
                    msdval /= syst.N

                    open(filenames[2], "a") do io
                        println(io, "$(time_acc),$(msdval)")
                    end

                    savemsd(hst_positions, filenames[3])
                end
            end

            if logging
                # * Save snapshot of the system
                snapshot(Array(positions), filenames[4], i)
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
