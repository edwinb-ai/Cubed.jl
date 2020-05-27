module Cubed

using Random
using RandomNumbers.Xorshifts

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
    P::V
    rc::V
    rng::Random.AbstractRNG
    ener::V
end

function energy(pos::AbstractArray, syst::System, pot::Potential)
    total_energy = 0.0f0
    virial = 0.0f0

    @inbounds for i = 1:syst.N
        @fastmath for j = (i + 1):syst.N
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
                    ener = (pot.a / pot.temp) *
                        ((1.0f0/Δpos)^pot.λ - (1.0f0/Δpos)^(pot.λ - 1.0f0))
                    ener += 1.0f0 / pot.temp
                    total_energy += ener
                end
            end
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


# function move(
#     positions::AbstractArray,
#     syst::System,
#     cycles::Int;
#     volume::Bool = false,
#     filename = nothing,
# )
#     # Run values
#     attempts = 0
#     volatt = 0
#     accepted = 0
#     volaccpt = 0
#     total_energy = 0
#     total_pressure = 0
#     total_ρ = 0
#     ratio = 0
#     volratio = 0
#     samples = 0

#     # Equilibration steps
#     @showprogress for i = 1:cycles
#         # * Attempt to move the particles around
#         attempts += 1
#         mc = metropolis!(positions, syst, disp)
#         accepted += mc
#         ratio = accepted / attempts

#         # * Attempt to change the volume in the system
#         if volume
#             if i % Int(syst.N * 2) == 0
#                 volatt += 1
#                 mc = mcvolume!(positions, syst, disp)
#                 volaccpt += mc
#                 volratio = volaccpt / volatt
#             end
#         end

#         # * Save to file
#         if i % syst.N == 0
#             samples += 1

#             # * Update the total energy of the system
#             total_energy += syst.ener

#             # * Update the total pressure of the system
#             total_pressure += syst.press / (syst.vol * 3.0)
#         end
#     end
#     # Save previous values
#     syst.ener = total_energy / samples
#     syst.press = total_pressure / samples

#     # Adjust results as averages
#     total_energy /= (syst.N * samples)
#     pressure = (total_pressure / samples) + (syst.ρ / syst.β)

#     # * Show results
#     println("Energy: $(total_energy)")
#     println("Ratio of acceptance: $(ratio)")
#     println("Pressure: $(pressure)")
# end

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
    positions = fill(0.0f0, 3, N)
    # Initialize the positions as grid
    init!(positions, syst)
    (syst.ener, _) = energy(positions, syst, potential)
    println("Initial energy: $(syst.ener / syst.N)")

    # * Main loop
    # Equilibration steps
    # move(positions, syst, 200000, dispm; volume = false)

    # # Sampling steps
    # move(positions, syst, 300000, dispm; volume = false, filename = "nvt_$(ρ)_$(P).dat")
end

run()

end # module
