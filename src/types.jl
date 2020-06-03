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
    press::AbstractArray
    rc::V
    rng::Random.AbstractRNG
    ener::AbstractArray
end