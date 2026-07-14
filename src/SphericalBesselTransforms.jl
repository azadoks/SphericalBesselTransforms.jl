module SphericalBesselTransforms

using LinearAlgebra
using Bessels: sphericalbesselj
using SpecialFunctions: gamma
using FFTW

export SBTPlan
export sbt!
export sbt
export sbtfreq

# `Base.logrange` only exists from Julia 1.11 on, and we still support 1.10 (the LTS).
_logrange(lo, hi, n) = exp.(range(log(lo), log(hi), n))

@inbounds function construct_Mℓ_t(ℓmax, nr, Δt, κmin, ρmin, Δρ)
    t = collect(0:(nr - 1)) * Δt
    Mℓ_t = zeros(Complex{Float64}, nr, ℓmax + 1)

    # ℓ=0
    # Eq.(19)
    Φ₁ = imag.(log.(gamma.(1 / 2 .- im * t)))
    Φ₂ = -atan.(tanh.(π * t / 2))
    Mℓ_t[:, 1] .= sqrt(π / 2) * exp.(im * (Φ₁ + Φ₂)) / nr
    Mℓ_t[1, 1] /= 2
    Mℓ_t[:, 1] .*= exp.(im * t * (κmin + ρmin))

    # ℓ=1
    # Eq.(16)
    if ℓmax >= 1
        ϕ = atan.(tanh.(π * t / 2)) - atan.(2t)
        Mℓ_t[:, 2] .= exp.(2im * ϕ) .* Mℓ_t[:, 1]
    end

    # ℓ=2...ℓmax
    # Eq.(24)
    for ℓ in 2:ℓmax
        iℓ = ℓ + 1
        ϕ = atan.(2t / (2 * (ℓ - 1) + 1))
        Mℓ_t[:, iℓ] .= exp.(-2im * ϕ) .* Mℓ_t[:, iℓ - 2]
    end

    # Eq.(35)
    ℓs = 0:ℓmax
    xs = exp.(ρmin .+ κmin .+ collect(0:(2nr - 1)) .* Δρ)
    jℓ_x = [sphericalbesselj(ℓ, x) for x in xs, ℓ in ℓs]
    M̄ℓ_t = conj.(ifft(jℓ_x, 1)[1:(nr + 1), :])
    return (; Mℓ_t, M̄ℓ_t)
end

"""
Return the `k` values corresponding to a logarithmic grid `r`.
"""
function sbtfreq(r, kmax = 500)
    if !all(r .≈ _logrange(minimum(r), maximum(r), length(r)))
        error("Input r must be logarithmically spaced")
    end
    ρmin = log(minimum(r))
    ρmax = log(maximum(r))
    Δρ = (ρmax - ρmin) / (length(r) - 1)
    κmax = log(kmax)
    κmin = κmax - (ρmax - ρmin)
    return exp(κmin) * exp.(collect(0:(length(r) - 1)) .* Δρ)
end

struct SBTPlan{T}
    ℓmax::Int
    r::Vector{T}
    rmin::T
    nr::Int
    Δρ::T
    k::Vector{T}
    kmin::T
    rext::Vector{T}
    post_div_fac::Vector{T}
    Mℓ_t::Matrix{Complex{T}}
    M̄ℓ_t::Matrix{Complex{T}}
    r32::Vector{T} # r^(3/2)
    rmin32::T # rmin^(3/2)
    rext32::Vector{T} # rext^(3/2)
    large_k_rfft_plan
    large_k_ifft_plan
    small_k_rfft_plan
    small_k_irfft_plan
    rfft_result_cache::Vector{Complex{T}}
    ifft_result_cache::Vector{Complex{T}}
    irfft_result_cache::Vector{T}
    frα_cache::Vector{T}
    yeM_cache::Vector{Complex{T}}
    g_cache::Vector{T}
end

function SBTPlan{T}(r::AbstractVector{T}, ℓmax::Int = 10, kmax::T = 500) where {T}
    if !all(r .≈ _logrange(minimum(r), maximum(r), length(r)))
        error("Input r must be logarithmically spaced")
    end
    nr = length(r)
    rmin = minimum(r)

    ρmin = log(minimum(r))
    ρmax = log(maximum(r))
    Δρ = (ρmax - ρmin) / (nr - 1)

    κmax = log(kmax)
    κmin = κmax - (ρmax - ρmin)
    k = exp(κmin) * exp.(collect(0:(nr - 1)) .* Δρ)
    kmin = minimum(k)

    rext = minimum(r) * exp.(collect(-nr:-1) .* Δρ)
    post_div_fac = exp.(-collect(0:(nr - 1)) * 3 / 2 * Δρ)

    Δt = 2π / (2nr * Δρ)
    Mℓ_t, M̄ℓ_t = construct_Mℓ_t(ℓmax, nr, Δt, κmin, ρmin, Δρ)

    # Profiling showed that these are very expensive to compute
    # in the sbt function, so we precompute them here.
    r32 = r .^ (3 / 2)
    rmin32 = rmin^(3 / 2)
    rext32 = rext .^ (3 / 2)

    large_k_rfft_plan = plan_rfft(zeros(T, 2nr))
    large_k_ifft_plan = plan_ifft(zeros(Complex{T}, 2nr))

    small_k_rfft_plan = plan_rfft(zeros(T, 2nr))
    small_k_irfft_plan = plan_irfft(zeros(Complex{T}, nr + 1), 2nr)

    rfft_result_cache = zeros(Complex{T}, nr + 1)
    ifft_result_cache = zeros(Complex{T}, 2nr)
    irfft_result_cache = zeros(T, 2nr)

    frα_cache = zeros(T, 2nr)
    yeM_cache = zeros(Complex{T}, 2nr)
    g_cache = zeros(T, nr)

    return SBTPlan{T}(
        ℓmax,
        r, rmin, nr, Δρ,
        k, kmin,
        rext, post_div_fac,
        Mℓ_t, M̄ℓ_t,
        r32, rmin32, rext32,
        large_k_rfft_plan, large_k_ifft_plan,
        small_k_rfft_plan, small_k_irfft_plan,
        rfft_result_cache, ifft_result_cache, irfft_result_cache,
        frα_cache, yeM_cache, g_cache
    )
end

function SBTPlan(r::AbstractVector, ℓmax = 10, kmax = 500)
    T = promote_type(eltype(r), typeof(kmax))
    return SBTPlan{T}(convert.(T, r), ℓmax, convert(T, kmax))
end

@inbounds function sbt_large_k!(
        g::AbstractVector{T},
        ℓ::Integer,
        f::AbstractVector{T},
        plan::SBTPlan{T},
        np_in::Integer,
        direction::Symbol,
        normalize::Bool
    ) where {T}
    # Follows the procedure outlined in Sec.(3) of the paper following Eq.(32)
    sqrt_2_over_π = normalize ? sqrt(2 / T(π)) : T(1)
    if direction == :forward
        rmin = plan.rmin
        kmin = plan.kmin
        C = f[1] / plan.rmin^(np_in + ℓ)
    elseif direction == :inverse
        rmin = plan.kmin
        kmin = plan.rmin
        C = f[1] / plan.kmin^(np_in + ℓ)
    else
        error("Invalid direction: $direction")
    end
    # Step 1
    for i in 1:plan.nr
        plan.frα_cache[i] = C * plan.rext[i]^(np_in + ℓ) * plan.rext32[i] / plan.rmin32
    end
    for i in 1:plan.nr
        plan.frα_cache[plan.nr + i] = f[i] * plan.r32[i] / plan.rmin32
    end

    # Step 2
    mul!(plan.rfft_result_cache, plan.large_k_rfft_plan, plan.frα_cache)

    # Step 3
    for i in 1:plan.nr
        plan.yeM_cache[i] = conj(plan.rfft_result_cache[i]) * plan.Mℓ_t[i, ℓ + 1]
    end

    # Steps 4 and 5
    mul!(plan.ifft_result_cache, plan.large_k_ifft_plan, plan.yeM_cache)
    c = (rmin / kmin)^(3 / 2) * 2plan.nr * sqrt_2_over_π
    for i in 1:plan.nr
        g[i] = c * real(plan.ifft_result_cache[plan.nr + i]) * plan.post_div_fac[i]
    end
    return g
end

function sbt_large_k(
        ℓ::Integer,
        f::AbstractVector{T},
        plan::SBTPlan{T},
        np_in::Integer,
        direction::Symbol,
        normalize::Bool
    ) where {T}
    g = zeros(T, plan.nr)
    return sbt_large_k!(g, ℓ, f, plan, np_in, direction, normalize)
end

@inbounds function sbt_small_k!(
        g::AbstractVector{T},
        ℓ::Integer,
        f::AbstractVector{T},
        plan::SBTPlan{T},
        direction::Symbol,
        normalize::Bool
    ) where {T}
    sqrt_2_over_π = normalize ? sqrt(2 / T(π)) : T(1)
    if direction == :forward
        plan.frα_cache[1:plan.nr] .= plan.r .^ 3 .* f
    elseif direction == :inverse
        plan.frα_cache[1:plan.nr] .= plan.k .^ 3 .* f
    else
        error("Invalid direction: $direction")
    end
    plan.frα_cache[(plan.nr + 1):end] .= T(0)
    mul!(plan.rfft_result_cache, plan.small_k_rfft_plan, plan.frα_cache)
    for i in 1:plan.nr + 1
        plan.yeM_cache[i] = (
            conj(plan.rfft_result_cache[i]) * plan.M̄ℓ_t[i, ℓ + 1] * sqrt_2_over_π
        )
    end
    mul!(
        plan.irfft_result_cache,
        plan.small_k_irfft_plan,
        view(plan.yeM_cache, 1:(plan.nr + 1))
    )
    for i in 1:plan.nr
        g[i] = real(plan.irfft_result_cache[i]) * 2plan.nr * plan.Δρ
    end
    return g
end

function sbt_small_k(
        ℓ::Integer,
        f::AbstractVector{T},
        plan::SBTPlan{T},
        direction::Symbol,
        normalize::Bool
    ) where {T}
    g = zeros(T, plan.nr)
    return sbt_small_k!(g, ℓ, f, plan, direction, normalize)
end

function sbt!(
        g::AbstractVector{T},
        ℓ::Integer,
        f::AbstractVector{T},
        plan::SBTPlan{T},
        np_in::Integer = 1; normalize = false,
        direction = :forward
    ) where {T}
    sbt_large_k!(g, ℓ, f, plan, np_in, direction, normalize)
    sbt_small_k!(plan.g_cache, ℓ, f, plan, direction, normalize)
    minloc = argmin(1:plan.nr) do i
        gi_large::Float64 = g[i]
        gi_small::Float64 = plan.g_cache[i]
        abs(gi_large - gi_small)
    end
    copy!(view(g, 1:minloc), view(plan.g_cache, 1:minloc))
    return g
end

function sbt(
        ℓ::Integer,
        f::AbstractVector{T},
        plan::SBTPlan{T},
        np_in::Integer = 1; normalize = false,
        direction = :forward
    ) where {T}
    g = zeros(T, plan.nr)
    return sbt!(g, ℓ, f, plan, np_in; normalize=normalize, direction=direction)
end

function sbt(
        ℓ::Integer,
        f::AbstractVector{T},
        r::AbstractVector{T};
        kmax = 500, np_in = 1,
        kwargs...
    ) where {T}
    plan = SBTPlan{T}(r, ℓ, convert(T, kmax))
    return sbt(ℓ, f, plan, np_in; kwargs...), plan.k
end

end # module SphericalBesselTransforms
