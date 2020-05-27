module Cubed

using Random
using RandomNumbers.Xorshifts
using ProgressMeter

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

function energy!(pos, forces, syst, pot)
    total_energy = 0.0f0
    virial = 0.0f0
    force = 0.0f0
    ener = 0.0f0

    @inbounds for i = 1:syst.N-1
        for j = (i + 1):syst.N
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
                    force *= (pot.a / pot.temp)
                else
                    force = 0.0f0
                    ener = 0.0f0
                end
            end
            # * Update the energy
            total_energy += ener

            # * Update the forces
            forces[1, i] += (force * xij) / Δpos
            forces[1, j] -= (force * xij) / Δpos

            forces[2, i] += (force * yij) / Δpos
            forces[2, j] -= (force * yij) / Δpos

            forces[3, i] += (force * zij) / Δpos
            forces[3, j] -= (force * zij) / Δpos

            # * Compute the virial
            virial += (force * xij * xij) + (force * yij * yij) + (force * zij * zij)
            virial /= Δpos
        end
    end

    return (total_energy, virial)
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
    for j  = axes(positions, 2)
        @inbounds for i = axes(positions, 1)
            positions[i, j] += (forces[i, j] * dynamics.τ) + rnd_matrix[i, j]
            if pbc
                positions[i, j] -= syst.L * round(positions[i, j] / syst.L)
            end
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

    rnd_matrix = Matrix{Float32}(undef, 3, syst.N)

    # Equilibration steps
    @showprogress for i = 1:cycles
        # * Create array of random numbers
        randn!(syst.rng, rnd_matrix)
        rnd_matrix .*= sqrt(2.0f0 * dynamics.τ)

        # * Attempt to move the particles around
        attempts += 1
        ermak!(positions, forces, syst, dynamics, rnd_matrix)

        # ! Initialize forces
        fill!(forces, 0.0f0)
        (syst.ener, syst.press) = energy!(positions, forces, syst, potential)

        # * Save to file
        if i % syst.N == 0
            samples += 1

            # * Update the total energy of the system
            total_energy += syst.ener

            # * Update the total pressure of the system
            total_pressure += syst.press / (syst.vol * 3.0)

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
    N = 256  # number of particles
    ρ = 0.76f0 # reduced density
    volumen = N / ρ # reduced volume
    Lcaja = ∛volumen # Cubic box length
    T = 1.0f0 # Reduced temperature
    rc = Lcaja / 2.0f0
    P = 0.0f0

    # Create a rng
    rng = Xoroshiro128Star(123456)
    # Initialize the system object
    syst = System(N, Lcaja, ρ, volumen, 1.0f0 / T, P, rc, rng, 0.0f0)
    dynamic = Dynamic(0.00005f0)
    λ = 50
    b = convert(Float32, λ / (λ - 1.0f0))
    a = convert(Float32, λ * b^(λ - 1.0f0))
    potential = Potential(a, b, 1.4737f0, 50)

    # Create the positions vector
    positions = Matrix{Float32}(undef, 3, N)
    fill!(positions, 0.0f0)
    forces = Matrix{Float32}(undef, 3, N)
    fill!(forces, 0.0f0)
    # Initialize the positions as grid
    init!(positions, syst)
    (syst.ener, syst.press) = energy!(positions, forces, syst, potential)
    println("Initial energy: $(syst.ener / syst.N)")

    # * Main loop
    # Equilibration steps
    move(positions, forces, syst, dynamic, potential, 200000)

    # # Sampling steps
    # move(positions, syst, 300000, dispm; volume = false, filename = "nvt_$(ρ)_$(P).dat")
end

run()

end # module
