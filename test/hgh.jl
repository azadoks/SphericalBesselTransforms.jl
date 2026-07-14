const HGH_FAMILIES = (
    "cp2k.nc.sr.lda.v0_1.largecore.gth",
    "cp2k.nc.sr.lda.v0_1.semicore.gth",
    "cp2k.nc.sr.lda.v0_1.smallcore.gth"
)

function hgh_local_direct(file, r::T) where {T <: Real}
    r == 0 && return hgh_local_direct(file, eps(T)) # quick hack for the division by zero below
    cloc::Vector{T} = [convert.(T, file.cloc); zeros(T, 4 - length(file.cloc))]
    rr = r / file.rloc
    convert(T,
        - sum(file.zion) / r * erf(rr / sqrt(T(2)))
        + exp(-rr^2 / 2) * (cloc[1] + cloc[2] * rr^2 + cloc[3] * rr^4 + cloc[4] * rr^6)
    )
end
function hgh_local_polynomial(T, file, t=Polynomial(T[0, 1]))
    cloc::Vector{T} = [convert.(T, file.cloc); zeros(T, 4 - length(file.cloc))]
    rloc::T = file.rloc
    Zion::T = sum(file.zion)
    # The polynomial prefactor P(t) (as used inside the { ... } brackets of equation
    # (5) of the HGH98 paper)
    P = (  cloc[1]
         + cloc[2] * (  3 -    t^2              )
         + cloc[3] * ( 15 -  10t^2 +   t^4      )
         + cloc[4] * (105 - 105t^2 + 21t^4 - t^6))

    4T(π) * rloc^2 * (-Zion + sqrt(T(π) / 2) * rloc * t^2 * P)
end
function hgh_local_fourier(file, p::T) where {T}
    t::T = p * file.rloc
    hgh_local_polynomial(T, file, t) * exp(-t^2 / 2) / t^2
end
function hgh_projector_direct(file::HghFile, i, l, r::T) where {T <: Real}
    rp = T(file.rp[l + 1])
    ired = (4i - 1) / T(2)
    sqrt(T(2)) * r^(l + 2(i - 1)) * exp(-r^2 / 2rp^2) / rp^(l + ired) / sqrt(gamma(l + ired))
end
function hgh_projector_polynomial(T, file::HghFile, i, l, t=Polynomial(T[0, 1]))
    @assert 0 <= l <= length(file.rp) - 1
    @assert i > 0
    rp::T = file.rp[l + 1]
    common::T = 4T(π)^(5 / T(4)) * sqrt(T(2^(l + 1)) * rp^3)

    # Note: In the (l == 0 && i == 2) case the HGH paper has an error.
    #       The first 8 in equation (8) should not be under the sqrt-sign
    #       This is the right version (as shown in the GTH paper)
    (l == 0 && i == 1) && return convert(typeof(t), common)
    (l == 0 && i == 2) && return common * 2 /  sqrt(T(  15))       * ( 3 -   t^2      )
    (l == 0 && i == 3) && return common * 4 / 3sqrt(T( 105))       * (15 - 10t^2 + t^4)
    #
    (l == 1 && i == 1) && return common * 1 /  sqrt(T(   3)) * t
    (l == 1 && i == 2) && return common * 2 /  sqrt(T( 105)) * t   * ( 5 -   t^2)
    (l == 1 && i == 3) && return common * 4 / 3sqrt(T(1155)) * t   * (35 - 14t^2 + t^4)
    #
    (l == 2 && i == 1) && return common * 1 /  sqrt(T(  15)) * t^2
    (l == 2 && i == 2) && return common * 2 / 3sqrt(T( 105)) * t^2 * ( 7 -   t^2)
    #
    (l == 3 && i == 1) && return common * 1 /  sqrt(T( 105)) * t^3

    error("Not implemented for l=$l and i=$i")
end
function hgh_projector_fourier(file::HghFile, i, l, p::T) where {T <: Real}
    t::T = p * file.rp[l + 1]
    hgh_projector_polynomial(T, file, i, l, t) * exp(-t^2 / 2)
end

function test_hgh(file::HghFile, plan)
    @testset "Vloc" begin
        Vloc = hgh_local_direct.(file, plan.r)
        Vloc_corr = Vloc .+ sum(file.zion) ./ plan.r
        V̄loc_true = hgh_local_fourier.(file, plan.k)
        V̄loc_true_corr = V̄loc_true .+ 4π * sum(file.zion) ./ plan.k.^2
        V̄loc_sbt = 4π * sbt(0, Vloc_corr, plan; normalize=false)
        @test all(isapprox.(V̄loc_sbt, V̄loc_true_corr, atol=1e-6))
    end
    @testset "β" begin
        for l in 0:file.lmax
            for i in 1:size(file.h[l+1], 1)
                P = hgh_projector_direct.(file, i, l, plan.r)
                P̄_true = hgh_projector_fourier.(file, i, l, plan.k)
                P̄_sbt = 4π * sbt(l, P, plan; normalize=false)
                @test all(isapprox.(P̄_sbt, P̄_true, atol=1e-10))
                P_sbt = sqrt(2/π) * sbt(
                    l, sqrt(2/π) .* P̄_sbt ./ (4π), plan;
                    direction=:inverse, normalize=false
                )
                @test all(isapprox.(P_sbt, P, atol=1e-5))
            end
        end
    end
end

@testset "HGH" begin
    N = 256
    rmin = 1.0 / 1024 / 32
    rmax = 20.0
    kmax = 500.0
    ℓmax = 4
    r = collect(_logrange(rmin, rmax, N))
    plan = SBTPlan{Float64}(r, ℓmax, kmax)
    for family_name in HGH_FAMILIES
        @testset "$(family_name)" begin
            family = PseudoFamily(family_name)
            for (el, path) in family
                @testset "$(el)" begin
                    file = HghFile(path)
                    test_hgh(file, plan)
                end
            end
        end
    end
end
