function gpu_energy!(
    pos::AbstractArray,
    forces::AbstractArray,
    N::Integer,
    L::Real,
    rc::Real,
    a::Real,
    b::Real,
    λ::Integer,
    temp::Real,
    full_ener::AbstractArray,
    vir::AbstractArray
)
    total_energy = 0.0f0
    virial = 0.0f0
    force = 0.0f0
    ener = 0.0f0

    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    @inbounds for i = index:stride:N
        # Reset variables
        total_energy = 0.0f0
        virial = 0.0f0

        for j = 1:N

            if i == j
                continue
            end

            xij = pos[1, i] - pos[1, j]
            yij = pos[2, i] - pos[2, j]
            zij = pos[3, i] - pos[3, j]

            # Periodic boundaries
            xij -= L * round(xij / L)
            yij -= L * round(yij / L)
            zij -= L * round(zij / L)

            # Compute distance
            Δpos = xij * xij + yij * yij + zij * zij
            Δpos = CUDA.sqrt(Δpos)

            if Δpos < rc
                if Δpos < b
                    # * Energy computation
                    ener = CUDA.pow(1.0f0 / Δpos, λ) - CUDA.pow(1.0f0 / Δpos, λ - 1.0f0)
                    ener *= a / temp
                    ener += 1.0f0 / temp

                    # * Force computation
                    force = λ * CUDA.pow(1.0f0 / Δpos, λ + 1.0f0)
                    force -= (λ - 1.0f0) * CUDA.pow(1.0f0 / Δpos, λ)
                    force *= a / temp
                else
                    force = 0.0f0
                    ener = 0.0f0
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
                virial += (force * xij * xij / Δpos) + (force * yij * yij / Δpos)
                virial += (force * zij * zij / Δpos)
            end
        end

        # * Save the values in their respective arrays
        full_ener[i] = total_energy
        vir[i] = virial
    end

    return nothing
end

function gpu_ermak!(
    positions::AbstractArray,
    forces::AbstractArray,
    N::Integer,
    L::Real,
    τ::Real,
    rnd_matrix::AbstractArray;
    pbc::Bool = true
)

    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    for j = index:stride:N
        positions[1, j] += (forces[1, j] * τ) + rnd_matrix[1, j]
        positions[2, j] += (forces[2, j] * τ) + rnd_matrix[2, j]
        positions[3, j] += (forces[3, j] * τ) + rnd_matrix[3, j]

        if pbc
            positions[1, j] -= L * round(positions[1, j] / L)
            positions[2, j] -= L * round(positions[2, j] / L)
            positions[3, j] -= L * round(positions[3, j] / L)
        end
    end

    return nothing
end
