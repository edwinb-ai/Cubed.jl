function msd!(positions::AbstractArray, ref_pos::AbstractArray)
    result = 0.0f0

    @inbounds for i = axes(positions, 2)
        δx = positions[1, i] - ref_pos[1, i]
        δy = positions[2, i] - ref_pos[2, i]
        δz = positions[3, i] - ref_pos[3, i]

        result += δx*δx + δy*δy + δz*δz
    end

    return result
end

function savemsd(positions::AbstractArray, path::String)

    @inbounds for i ∈ axes(positions, 2)
        filename = joinpath(path, "msd_$(i).csv")

        open(filename, "a") do io
            println(io, "$(positions[1, i]),$(positions[2, i]),$(positions[3, i])")
        end
    end

    return nothing
end