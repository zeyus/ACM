using CSV, DataFrames
using Optim, Turing
using FillArrays, Distributions
using StatsFuns
using Plots, StatsPlots
using Random
using LinearAlgebra
using Turing: ForwardDiff
using ReverseDiff


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
N = 120
hand_p = 0.70

# let's run a bunch of simulations
nsims = 10
true_values = zeros(nsims, 3)
prior_chains = Dict{Int, DataFrame}()
posterior_chains = Dict{Int, DataFrame}()
for i in 1:nsims
    rh = rand(Bernoulli(hand_p), N)
    lh = 1 .- rh
    reward = [lh rh]


    h_missing = Vector{Union{Missing, Int}}(missing, N)
    h, true_α, true_τ = RWCoins(h_missing; N=N, reward=reward)()
    true_values[i, 1] = true_α
    true_values[i, 2] = true_τ
    true_values[i, 3] = hand_p
    chains_prior = sample(RWCoins(h_missing; N=N, reward=reward), Prior(), 1_000)
    chains_prior_df = DataFrame(chains_prior)
    prior_chains[i] = chains_prior_df
    # parameter recovery
    model = RWCoins(h; N=N, reward=reward)
    chains = sample(model, NUTS(1_000, 0.99; max_depth=20, adtype=Turing.AutoReverseDiff(true)), MCMCThreads(), 1_000, 4)
    # single thread for debugging
    # chains = sample(model, NUTS(1_000, 0.99; max_depth=20, adtype=Turing.AutoReverseDiff(true)), 3_000)

    chain_df = DataFrame(chains)
    posterior_chains[i] = chain_df
    # print("True α:", true_α, ", True τ:", true_τ, "\n")

    # density(chain_df[!, :α], label="Posterior α", xlabel="value", ylabel="Frequency", title="Parameters", color=:purple, fill=(0, 0.3))
    # density!(chain_df[!, :τ], label="Posterior τ", color=:blue, fill=(0, 0.3))
    # # add priors 
    # density!(chains_prior_df[!, :α], label="Prior α", color=:purple, linestyle=:dash)
    # density!(chains_prior_df[!, :τ], label="Prior τ", color=:blue, linestyle=:dash)
    # # add true values
    # vline!([true_α], label="True α", color=:purple, linestyle=:solid)
    # vline!([true_τ], label="True τ", color=:blue, linestyle=:solid)
end

# plot overall parameter recovery (points for true values, lines for posterior means)
# alpha

modes_α = []
modes_τ = []
for i in 1:nsims
    chain_df = posterior_chains[i]
    push!(modes_α, mode(chain_df[!, :α]))
    push!(modes_τ, mode(chain_df[!, :τ]))
end

# plot true vs mode for alpha
p1 = plot(dpi=300)
scatter!(true_values[:, 1], modes_α, label="True vs Mode α", xlabel="True α", ylabel="Mode α", title="True vs Mode α", color=:purple)
# add line
max_x = maximum([maximum(true_values[:, 1]), maximum(modes_α)])
max_y = maximum([maximum(true_values[:, 1]), maximum(modes_α)])
plot!([0, max_x], [0, max_y], label="y=x", color=:black)
# save to png
savefig(p1, "true_vs_mode_alpha.png")

# plot true vs mode for tau
p2 = plot(dpi=300)
scatter!(true_values[:, 2], modes_τ, label="True vs Mode τ", xlabel="True τ", ylabel="Mode τ", title="True vs Mode τ", color=:blue)
# add line
max_x = maximum([maximum(true_values[:, 2]), maximum(modes_τ)])
max_y = maximum([maximum(true_values[:, 2]), maximum(modes_τ)])
plot!([0, max_x], [0, max_y], label="y=x", color=:black)
# save to png
savefig(p2, "true_vs_mode_tau.png")

# plot true vs posterior Distributions
i = 2
p3 = plot(dpi=300)
chain_df = posterior_chains[i]
density!(chain_df[!, :α], label="Posterior α", xlabel="value", ylabel="Frequency", title="Parameters", color=:purple, fill=(0, 0.3))
density!(chain_df[!, :τ], label="Posterior τ", color=:blue, fill=(0, 0.3))
# add priors 
chains_prior_df = prior_chains[i]
density!(chains_prior_df[!, :α], label="Prior α", color=:purple, linestyle=:dash)
density!(chains_prior_df[!, :τ], label="Prior τ", color=:blue, linestyle=:dash)
# add true values
true_α = true_values[i, 1]
true_τ = true_values[i, 2]
vline!([true_α], label="True α", color=:purple, linestyle=:solid)
vline!([true_τ], label="True τ", color=:blue, linestyle=:solid)
xlims!(0, 2)

# save to pn
savefig(p3, "true_vs_posterior_dist.png")