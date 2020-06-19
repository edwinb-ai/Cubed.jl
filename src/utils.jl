"""
    Create a mesh-like initial configuration for the simulation system.
It uses the information from the system such as the cut-off radius and the
density of the system to allocate the particles.

It modifies the original `positions` array, so it is an in-place operation.
"""
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

function snapshot(positions, filename, i)
    snapshot_file = joinpath(filename, "system_$(i).csv")
    CSV.write(snapshot_file, DataFrame(positions))
    
    return nothing    
end
