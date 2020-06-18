struct Dynamic{V <: Real}
    τ::V
end

mutable struct Potential{V <: Real}
    a::V
    b::V
    temp::V
    λ::Integer
end

mutable struct System{V <: Real}
    N::Int
    L::V
    ρ::V
    vol::V
    press::AbstractArray
    rc::V
    rng::Random.AbstractRNG
    ener::AbstractArray
end

"""
    The system created should always allocate the pressure
and energy arrays for the user.
"""
function System(N, Lbox, ρ, volumen, rc, rng)
    P = Vector{Float32}(undef, N)
    E = Vector{Float32}(undef, N)
    
    return System{Float32}(N, Lbox, ρ, volumen, P, rc, rng, E)
end