using DataFrames
using Plots, StatsPlots
using HDF5
using MCMCChains
using MCMCChainsStorage
include("../LogCommon.jl")

@info "Reading chains..."
chain_file = "./chains/sim_chains.h5"
restored_chains::Dict{String, Chains} = Dict()
chains = h5open(chain_file, "r") do file
    for key in keys(file)
        @info "Reading $key"
        g = open_group(file, key)
        restored_chains[key] = read(g, Chains)
    end
end

@info "Plotting chains..."
# plot some
p = plot(restored_chains["posterior_10-50"], plot_title="Posterior α and τ (N trials = 1000)", dpi=300, size=(1000, 700))

# save the plot
savefig(p, "./out/posterior_10-50.png")

p = plot(restored_chains["posterior_1-50"], plot_title="Posterior α and τ (N trials = 10)", dpi=300, size=(1000, 700))

# save the plot
savefig(p, "./out/posterior_1-50.png")

Plots.getattr(p)[:size]