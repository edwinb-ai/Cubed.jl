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