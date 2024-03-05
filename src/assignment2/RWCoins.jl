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
# real_p is the real probability of the coin being in hand 1 (default is 0.8)
@model function RWCoins(h::Union{Missing, Vector{Union{Missing, Int}}}, ::Type{T} = Float64; N::Int, reward::Matrix{Int}) where T
    # Priors
    α ~ LogNormal(0, 0.3)  # learning rate
    β ~ LogNormal(0, 0.5)  # inverse temperature (noise/explore-exploit tradeoff)
    
    v = T(0.0)
    if ismissing(h)
        h = Vector{Union{Missing, Int}}(missing, N)
    end

    pₜ = logit(0.5)
    for t in 1:N
        h[t] ~ BernoulliLogit(pₜ)
        hₜ = ReverseDiff.value(h[t]) + 1
        v = v + α * (reward[t, hₜ] - logistic(pₜ))
        pₜ = logit(1 / (1 + exp(-β * v)))
    end
    return h, α, β
end
# read simdata/W3_randomnoise.csv into a dataframe
# df = DataFrame(CSV.File("simdata/W3_randomnoise.csv"))

# # only include noise == 0 and rate == 0.8
# df1 = filter(row -> row.noise == 0 && row.rate == 0.8, df)

# # convert the dataframe to a vector
# h = Vector{Int}(df1[!, :choice])
# N = length(h)

# run the model (sim)
N = 300
hand_p = logit(0.7)
rh = rand(BernoulliLogit(hand_p), N)
lh = 1 .- rh
reward = [lh rh]
# make zeros -1 (to punish loss)
# reward = 2 .* reward .- 1

h_missing = Vector{Union{Missing, Int}}(missing, N)
h, true_α, true_β = RWCoins(h_missing; N=N, reward=reward)()
chains_prior = sample(RWCoins(h_missing; N=N, reward=reward), Prior(), 5_000)
chains_prior_df = DataFrame(chains_prior)
# parameter recovery
model = RWCoins(h; N=N, reward=reward)
chains = sample(model, NUTS(1_000, 0.99; max_depth=20, adtype=Turing.AutoReverseDiff(true)), MCMCThreads(), 3_000, 3)
# single thread for debugging
# chains = sample(model, NUTS(1_000, 0.99; max_depth=20, adtype=Turing.AutoReverseDiff(true)), 3_000)

chain_df = DataFrame(chains)
print("True α:", true_α, ", True β:", true_β, "\n")

density(chain_df[!, :α], label="Posterior α", xlabel="value", ylabel="Frequency", title="Parameters", color=:purple, fill=(0, 0.3))
density!(chain_df[!, :β], label="Posterior β", color=:blue, fill=(0, 0.3))
# add priors 
density!(chains_prior_df[!, :α], label="Prior α", color=:purple, linestyle=:dash)
density!(chains_prior_df[!, :β], label="Prior β", color=:blue, linestyle=:dash)
# add true values
vline!([true_α], label="True α", color=:purple, linestyle=:solid)
vline!([true_β], label="True β", color=:blue, linestyle=:solid)

# set xlimit 
xlims!(0, 3)


