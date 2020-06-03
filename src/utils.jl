function init!(positions::AbstractArray, syst::System)
    n3 = 2
    ix = 0
    iy = 0
    iz = 0

    # Find the lowest perfect cube, n3, greater than or equal to the
    # number of particles
    while n3^3 < syst.N
        n3 += 1
    end

    for i in axes(positions, 2)
        positions[1, i] = (ix + 0.5f0) * syst.L / n3
        positions[2, i] = (iy + 0.5f0) * syst.L / n3
        positions[3, i] = (iz + 0.5f0) * syst.L / n3
        ix += 1

        if ix == n3
            ix = 0
            iy += 1
            if iy == n3
                iy = 0
                iz += 1
            end
        end
    end
end


function move(positions, forces, syst, dynamics, potential, cycles; filename = nothing)
    # * Accumulation variables
    total_energy = 0.0f0
    total_pressure = 0.0f0
    big_z = 0.0f0
    total_virial = 0.0f0
    samples = 0

    # GPU variables for the kernels
    nthreads = 256
    gpu_blocks = ceil(Int(syst.N / nthreads))

    # Allocate memory for random numbers
    rnd_matrix = Matrix{Float32}(undef, 3, syst.N)

    # ! Main loop
    @showprogress for i = 1:cycles
        # * Create array of random numbers
        randn!(syst.rng, rnd_matrix)
        rnd_matrix .*= sqrt(2.0f0 * dynamics.τ)
        cu_rand_matrix = CuArray(rnd_matrix)

        # * Move the particles following the Ermak-McCammon algorithm
        CUDA.@sync begin
            @cuda threads=nthreads blocks=gpu_blocks ermak!(positions,
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
            @cuda threads=nthreads blocks=gpu_blocks energy!(positions,
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
        if i >= 50000 && i % syst.N == 0
            samples += 1

            # * Update the total energy of the system
            total_energy += sum(syst.ener)

            # * Extract the virial from the kernel
            total_virial = sum(syst.press)
            total_virial /= 3.0f0
            big_z = 1.0f0 + (total_virial / syst.N)
            # * Update the total pressure of the system
            total_pressure = big_z / syst.ρ

            if !isnothing(filename)
                open(filename, "a") do io
                    filener = total_energy / (syst.N * samples)
                    filepress = (total_pressure / samples) + syst.ρ
                    println(io, "$(filener),$(filepress)")
                end
            end
        end
    end
    # Save previous values
    # syst.ener = total_energy / samples
    # syst.press = total_pressure / samples

    # Adjust results as averages
    total_energy /= (syst.N * samples)
    pressure = total_pressure / samples

    # * Show results
    println("Energy: $(total_energy)")
    println("Pressure: $(pressure)")
    println("Compressibility: $(big_z)")
end
