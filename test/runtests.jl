using Test
using PseudoPotentialData
using PseudoPotentialIO
using Polynomials
using SpecialFunctions

using SphericalBesselTransforms
using SphericalBesselTransforms: _logrange

@testset "SphericalBesselTransforms.jl" begin
    @testset "sanity" begin
        @test_throws "Input r must be log" sbtfreq(1.0:0.1:10.0)
        @test_throws "Input r must be log" SBTPlan{Float64}(1.0:0.1:10.0, 0, 10.0)
        plan = SBTPlan{Float64}(collect(_logrange(1e-5, 20, 10)), 0, 10.0)
        @test_throws "Invalid dir" sbt(0, zeros(10), plan, direction=:foo)
    end

    @testset "sbtfreq" begin
        for n in [5, 7, 9, 10, 100, 101, 102, 1000, 1013, 78934]
            k = sbtfreq(_logrange(1e-5, 20, n))
            plan = SBTPlan{Float64}(_logrange(1e-5, 20, n), 0, 500.0)
            @test all(k .== plan.k)
        end
    end

    @testset "β³/2 exp(-βr)" begin
        N = 256
        rmin = 1.0 / 1024 / 32
        rmax = 20.0
        r = collect(_logrange(rmin, rmax, N))
        k = sbtfreq(r)
        β = 2.0
        f_true = @. β^3 / 2 * exp(-β * r)
        g_true = @. sqrt(2/π) * β^4 / (k^2 + β^2)^2
        g_sbt, _ = sbt(0, f_true, r; normalize=true, direction=:forward)
        @test all(isapprox.(g_sbt, g_true, atol=1e-10))
        f_sbt, _ = sbt(0, g_sbt, r, normalize=true, direction=:inverse)
        @test all(isapprox.(f_sbt, f_true, atol=1e-1))  # TODO: this is pretty bad
    end
    include("hgh.jl")
end
