using Pkg; Pkg.activate("examples");
using PseudoPotentialData
using PseudoPotentialIO
using CairoMakie
using SphericalBesselTransforms
using SphericalBesselTransforms: _logrange

# Implementation of HGH pseudopotential quantities, stolen from DFTK.jl
include("hgh.jl")

family = PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth")
pspfile = HghFile(family[:Si])

r = collect(_logrange(1.0 / 1024 / 32, 20.0, 512))
plan = SBTPlan{Float64}(r, pspfile.lmax, 500.0)

begin
    fig = Figure()
    ax_r = Axis(fig[1,1], xlabel="r (Bohr)", ylabel="β(r)", xscale=log10)
    ax_k = Axis(fig[2,1], xlabel="k (1/Bohr)", ylabel="β(k)", xscale=log10)
    colors = [:black, :blue, :red, :green, :cyan, :magenta, :orange, :purple]
    icol = 1
    for l in 0:pspfile.lmax
        for i in 1:size(pspfile.h[l+1], 1)
            β_r = hgh_projector_direct.(pspfile, i, l, r)
            β_k = hgh_projector_fourier.(pspfile, i, l, plan.k)
            β_k_sbt = 4π * sbt(l, β_r, plan; normalize=false)
            β_r_sbt = sqrt(2/π) * sbt(
                l, sqrt(2/π) .* β_k_sbt ./ (4π), plan;
                direction=:inverse, normalize=false
            )
            lines!(ax_r, r, β_r, label="l=$l i=$i", color=colors[icol])
            scatter!(ax_r, r[begin:10:end], β_r_sbt[begin:10:end], color=colors[icol])
            lines!(ax_k, plan.k, β_k, label="l=$l i=$i", color=colors[icol])
            scatter!(ax_k, plan.k[begin:10:end], β_k_sbt[begin:10:end], color=colors[icol])
            icol += 1
        end
    end
    axislegend(ax_r)
    fig
end
