using CSV, DataFrames
using Optim, Turing
using FillArrays, Distributions
using StatsFuns
using Plots, StatsPlots
using Random
using LinearAlgebra
using Turing: ForwardDiff
using ReverseDiff
using HDF5
using MCMCChains
using MCMCChainsStorage
using CategoricalArrays
using Logging, LoggingExtras, Dates

# set up logging
const date_format = "yyyy-mm-dd HH:MM:SS"

timestamp_logger(logger) = TransformerLogger(logger) do log
  merge(log, (; message = "$(Dates.format(now(), date_format)) $(log.message)"))
end

ConsoleLogger(stdout, Logging.Info) |> timestamp_logger |> global_logger

@info "Preparing model..."
# Rescorla-Wagner model
# h is the result of selected hands (0 or 1)
# N is the number of trials
# reward is a matrix of size N x 2, where reward[t, 1] is the reward of hand 1 at trial t
# (for this experiment, only one will be 1 and the other will be 0)
@model function RWCoins(h::Union{Missing, Vector{Union{Missing, Int}}}, ::Type{T} = Float64; N::Int, reward::Matrix{Int}) where T
    # Priors
    α ~ LogitNormal(0, 1)  # learning rate
    τ ~ LogNormal(0, 0.7)  # inverse temperature (noise/explore-exploit tradeoff)
    
    v = T(0.0)
    if ismissing(h)
        h = Vector{Union{Missing, Int}}(missing, N)
    end

    pₜ = 0.5
    for t in 1:N
        h[t] ~ Bernoulli(pₜ)
        hₜ = ReverseDiff.value(h[t]) + 1
        v += α * (reward[t, hₜ] - v)
        pₜ = (1 / (1 + exp(-τ * v)))
    end
    return h, α, τ
end
# read simdata/W3_randomnoise.csv into a dataframe
# df = DataFrame(CSV.File("simdata/W3_randomnoise.csv"))

# # only include noise == 0 and rate == 0.8
# df1 = filter(row -> row.noise == 0 && row.rate == 0.8, df)

# # convert the dataframe to a vector
# h = Vector{Int}(df1[!, :choice])
# N = length(h)

# run the model (sim)
N_vals = [10, 25, 50, 75, 100, 150, 200, 250, 500, 1000]
N_count = length(N_vals)
hand_p = 0.70
nsims = 100
# N = 120
chain_file = "./chains/sim_chains.h5"

# let's run a bunch of simulations
true_values = zeros(N_count, nsims, 3)
posterior_chains = fill(DataFrame(), N_count, nsims)
prior_chains = fill(DataFrame(), N_count, nsims)

for (i, N) in enumerate(N_vals)
    @info "Running simulation $(i)"
    modes_α = []
    modes_τ = []
    for j in 1:nsims
        @info "Running simulation $(i)-$(j)"
        rh = rand(Bernoulli(hand_p), N)
        lh = 1 .- rh
        reward = [lh rh]
        h_missing = Vector{Union{Missing, Int}}(missing, N)
        h, true_α, true_τ = RWCoins(h_missing; N=N, reward=reward)()
        true_values[i, j, 1] = true_α
        true_values[i, j, 2] = true_τ
        true_values[i, j, 3] = hand_p

        @info "Sampling from priors..."
        chains_prior = nothing
        with_logger(NullLogger()) do
            chains_prior = sample(RWCoins(h_missing; N=N, reward=reward), Prior(), 1_000, progress=false, verbose=false)
        end
        chains_prior_df = DataFrame(chains_prior)
        prior_chains[i, j] = chains_prior_df
        # parameter recovery
        model = RWCoins(h; N=N, reward=reward)
        
        @info "Fitting model..."
        chains = nothing
        with_logger(NullLogger()) do
            chains = sample(model, NUTS(1_000, 0.99; max_depth=20, adtype=Turing.AutoReverseDiff(true)), MCMCThreads(), 1_000, 4, progress=false, verbose=false)
        end
        # single thread for debugging
        # chains = sample(model, NUTS(1_000, 0.99; max_depth=20, adtype=Turing.AutoReverseDiff(true)), 3_000)
        # save chains

        @info "Saving chains..."
        h5open(chain_file, "cw") do file
            g = create_group(file, "prior_$i-$j")
            write(g, chains_prior)
            g = create_group(file, "posterior_$i-$j")
            write(g, chains)
        end
        chain_df = DataFrame(chains)
        posterior_chains[i, j] = chain_df
        # save modes
        push!(modes_α, mode(chain_df[!, :α]))
        push!(modes_τ, mode(chain_df[!, :τ]))
        # save plots of prior + posterior
        @info "Saving plots..."

        p = plot(dpi=300)
        density!(chain_df[!, :α], label="Posterior α", xlabel="value", ylabel="Frequency", title="Parameters", color=:purple, fill=(0, 0.3))
        density!(chain_df[!, :τ], label="Posterior τ", color=:blue, fill=(0, 0.3))
        # add priors
        density!(chains_prior_df[!, :α], label="Prior α", color=:purple, linestyle=:dash)
        density!(chains_prior_df[!, :τ], label="Prior τ", color=:blue, linestyle=:dash)
        # add true values
        vline!([true_α], label="True α", color=:purple, linestyle=:solid)
        vline!([true_τ], label="True τ", color=:blue, linestyle=:solid)
        xlims!(0, 3)
        savefig(p, "./out/posterior_vs_prior_$i-$j.png")

    end
    @info "Saving plots for N=$(N)"
    # plot true vs mode for alpha
    p = plot(dpi=300)
    scatter!(true_values[i, :, 1], modes_α, label="True vs Mode α", xlabel="True α", ylabel="Mode α", title="True vs Mode α (N=$N)", color=:purple)
    # add line
    max_x = maximum([maximum(true_values[i, :, 1]), maximum(modes_α)])
    max_y = maximum([maximum(true_values[i, :, 1]), maximum(modes_α)])
    plot!([0, max_x], [0, max_y], label="y=x", color=:black)
    # save to png
    savefig(p, "./out/true_vs_mode_alpha_N$N.png")

    # plot true vs mode for tau
    p = plot(dpi=300)
    scatter!(true_values[i, :, 2], modes_τ, label="True vs Mode τ", xlabel="True τ", ylabel="Mode τ", title="True vs Mode τ (N=$N)", color=:blue)
    # add line
    max_x = maximum([maximum(true_values[i, :, 2]), maximum(modes_τ)])
    max_y = maximum([maximum(true_values[i, :, 2]), maximum(modes_τ)])
    plot!([0, max_x], [0, max_y], label="y=x", color=:black)
    # save to png
    savefig(p, "./out/true_vs_mode_tau_N$N.png")
end

@info "Done running simulations!"
# boxplot of true vs mode for alpha and tau
@info "Calculating parameter recovery values..."
recovered_values = zeros(N_count, 2, nsims)
for i in 1:N_count
    for j in 1:nsims
        recovered_values[i, 1, j] = mode(posterior_chains[i, j][!, :α]) - true_values[i, j, 1]
        recovered_values[i, 2, j] = mode(posterior_chains[i, j][!, :τ]) - true_values[i, j, 2]
    end
end
# make a dataframe of all the recovered values
xN_vals = repeat(string.(repeat(N_vals,2)), inner=nsims)
xN_vals = categorical(xN_vals)
levels!(xN_vals, string.(N_vals))
param = repeat(["α", "τ"], inner=N_count * nsims)
y_α = reshape(recovered_values[:, 1, :], N_count * nsims)
y_τ = reshape(recovered_values[:, 2, :], N_count * nsims)

df_recovered = DataFrame(N=xN_vals, param=param, value=vcat(y_α, y_τ))





# order by N

@info "Saving parameter recovery plots..."

p = plot(dpi=300)
hline!([0], label="y=0", color=:black, linestyle=:dash)
@df df_recovered groupedboxplot!(:N, :value, group=:param, fillalpha=0.7, side=:left, xlabel="Number of Trials", ylabel="mode - true", title="Parameter recovery of by number of trials", color=[:purple :orange])
savefig(p, "./out/parameter_recovery_violin.png")

# save df
@info "Saving parameter recovery dataframe..."
CSV.write("./out/parameter_recovery.csv", df_recovered)

# add true values to dataframe
true_values_α = reshape(true_values[:, :, 1], N_count * nsims)
true_values_τ = reshape(true_values[:, :, 2], N_count * nsims)
df_recovered[!, :true_value] = vcat(true_values_α, true_values_τ)

# save df
@info "Saving parameter recovery dataframe with true values..."
CSV.write("./out/parameter_recovery_w_true.csv", df_recovered)

@info "Done! Goodnight."