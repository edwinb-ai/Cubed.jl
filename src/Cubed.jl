module Cubed

using Random
using RandomNumbers.Xorshifts
using ProgressMeter
using CUDA

struct Dynamic{V<:Real}
    τ::V
end

mutable struct Potential{V<:Real}
    a::V
    b::V
    temp::V
    λ::Integer
end

mutable struct System{V<:Real}
    N::Int
    L::V
    ρ::V
    vol::V
    β::V
    press::V
    rc::V
    rng::Random.AbstractRNG
    ener::V
end

function energy!(pos, forces, syst, pot, total_energy, virial)
    total_energy = 0.0f0
    virial = 0.0f0
    force = 0.0f0
    ener = 0.0f0

    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    for i = 1:syst.N
        for j = index:stride:syst.N
            xij = pos[1, i] - pos[1, j]
            yij = pos[2, i] - pos[2, j]
            zij = pos[3, i] - pos[3, j]

            # Periodic boundaries
            xij -= syst.L * round(xij / syst.L)
            yij -= syst.L * round(yij / syst.L)
            zij -= syst.L * round(zij / syst.L)

            # Compute distance
            Δpos = xij * xij + yij * yij + zij * zij
            Δpos = sqrt(Δpos)

            if Δpos < syst.rc
                if Δpos < pot.b
                    # * Energy computation
                    ener = (pot.a / pot.temp) *
                        ((1.0f0/Δpos)^pot.λ - (1.0f0/Δpos)^(pot.λ - 1.0f0))
                    ener += 1.0f0 / pot.temp


                    # * Force computation
                    force = pot.λ * (1.0f0/Δpos)^(pot.λ + 1.0f0)
                    force -= (pot.λ - 1.0f0) * (1.0f0/Δpos)^pot.λ
                    force *= pot.a / pot.temp
                else
                    force = 0.0f0
                    ener = 0.0f0
                end
            end
            # * Update the energy
            total_energy += ener

            # * Update the forces
            forces[1, i] += (force * xij) / Δpos
            forces[2, i] += (force * yij) / Δpos
            forces[3, i] += (force * zij) / Δpos

            forces[1, j] -= (force * xij) / Δpos
            forces[2, j] -= (force * yij) / Δpos
            forces[3, j] -= (force * zij) / Δpos

            # * Compute the virial
            virial += (force * xij * xij) + (force * yij * yij) + (force * zij * zij)
            virial *= 3.0f0 / Δpos
        end
    end

    # return (total_energy, virial)
    return nothing
end

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

function ermak!(positions, forces, syst, dynamics, rnd_matrix; pbc::Bool = true)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    for j = index:stride:syst.N
        positions[1, j] += (forces[1, j] * dynamics.τ) + rnd_matrix[1, j]
        positions[2, j] += (forces[2, j] * dynamics.τ) + rnd_matrix[2, j]
        positions[3, j] += (forces[3, j] * dynamics.τ) + rnd_matrix[3, j]

        if pbc
            positions[1, j] -= syst.L * round(positions[1, j] / syst.L)
            positions[2, j] -= syst.L * round(positions[2, j] / syst.L)
            positions[3, j] -= syst.L * round(positions[3, j] / syst.L)
        end
    end
end


function move(positions, forces, syst, dynamics, potential, cycles; filename = nothing)
    # Run values
    attempts = 0
    total_energy = 0
    total_pressure = 0
    total_ρ = 0
    samples = 0

    nthreads = 1
    gpu_blocks = syst.N / nthreads

    rnd_matrix = Matrix{Float32}(undef, 3, syst.N)

    # Equilibration steps
    @showprogress for i = 1:cycles
        # * Create array of random numbers
        randn!(syst.rng, rnd_matrix)
        rnd_matrix .*= sqrt(2.0f0 * dynamics.τ)
        cu_rand_matrix = CuArray(rnd_matrix)

        # * Attempt to move the particles around
        attempts += 1
        CUDA.@sync begin
            @cuda threads=nthreads blocks=gpu_blocks ermak!(positions,
                forces,
                syst,
                dynamics,
                cu_rand_matrix
            )
        end
        # display(forces)

        # ! Initialize forces
        CUDA.fill!(forces, 0.0f0)
        # (syst.ener, syst.press) = energy!(positions, forces, syst, potential)
        CUDA.@sync begin
            @cuda threads=nthreads blocks=gpu_blocks energy!(positions,
                forces,
                syst,
                potential,
                syst.ener,
                syst.press
            )
        end

        # * Save to file
        if i % syst.N == 0
            samples += 1

            # * Update the total energy of the system
            total_energy += syst.ener

            # * Update the total pressure of the system
            total_pressure += syst.press / (syst.vol * 3.0f0)

            if !isnothing(filename)
                open(filename, "a") do io
                    filener = total_energy / (syst.N * samples)
                    filepress = (total_pressure / samples) + (syst.ρ / syst.β)
                    println(io, "$(filener),$(filepress)")
                end
            end
        end
    end
    # Save previous values
    syst.ener = total_energy / samples
    syst.press = total_pressure / samples

    # Adjust results as averages
    total_energy /= (syst.N * samples)
    pressure = (total_pressure / samples) + (syst.ρ / syst.β)

    # * Show results
    println("Energy: $(total_energy)")
    println("Pressure: $(pressure)")
end

function run()
    # * Simulation parameters
    N = 2048  # number of particles
    ρ = 0.76f0 # reduced density
    volumen = N / ρ # reduced volume
    Lcaja = ∛volumen # Cubic box length
    T = 1.0f0 # Reduced temperature
    rc = Lcaja / 2.0f0
    P = 0.0f0

    # gpu_blocks = N / 256
    gpu_blocks = 1
    nthreads = 1

    # Create a rng
    rng = Xoroshiro128Star(123456)
    # Initialize the system object
    syst = System(N, Lcaja, ρ, volumen, 1.0f0 / T, P, rc, rng, 0.0f0)
    dynamic = Dynamic(0.000005f0)
    λ = 50
    b = convert(Float32, λ / (λ - 1.0f0))
    a = convert(Float32, λ * b^(λ - 1.0f0))
    potential = Potential(a, b, 1.4737f0, 50)

    # Create the positions vector
    positions = CuArray{Float32}(undef, 3, N)
    fill!(positions, 0.0f0)
    forces = CuArray{Float32}(undef, 3, N)
    fill!(forces, 0.0f0)
    # Initialize the positions as grid
    init!(positions, syst)
    # (syst.ener, syst.press) = energy!(positions, forces, syst, potential)
    CUDA.@sync begin
        @cuda threads=nthreads blocks=gpu_blocks energy!(positions,
            forces,
            syst,
            potential,
            syst.ener,
            syst.press
        )
    end
    println("Initial energy: $(syst.ener / syst.N)")

    # * Main loop
    # Equilibration steps
    move(positions, forces, syst, dynamic, potential, 10000)

    # # Sampling steps
    # move(positions, syst, 300000, dispm; volume = false, filename = "nvt_$(ρ)_$(P).dat")
end

run()

end # module
