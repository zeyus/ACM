using CSV, DataFrames
using Optim, Turing
using FillArrays, Distributions
using StatsFuns
using Plots, StatsPlots
using Random
using LinearAlgebra
using Turing: ForwardDiff


function noisy_softmax(x::Vector{Float64}, β::Float64)
    u = maximum(x)
    xt = x .- u
    lmul!(inv(sum(xt)), xt)
    return xt
end


# Rescorla-Wagner model
# h is the result of selected hands (0 or 1)
# N is the number of trials
# reward is a matrix of size N x 2, where reward[t, 1] is the reward of hand 1 at trial t
# (for this experiment, only one will be 1 and the other will be 0)
# real_p is the real probability of the coin being in hand 1 (default is 0.8)
@model function RWCoins(h::Union{Missing, Vector{Union{Missing, Int}}}, ::Type{T} = Float64; N::Int, reward::Matrix{Int}) where T
    # Priors
    α ~ Normal(0, 0.3)  # learning rate
    β ~ LogNormal(0, 0.5)  # inverse temperature (noise/explore-exploit tradeoff)
    

    v = Vector{Union{Float64}}([0.0, 0.0])
    if ismissing(h)
        h = Vector{Union{Missing, Int}}(missing, N)
    end


    pred_err::Float64 = 0.0
    p::Vector{Float64} = [0.5, 0.5]
    for t in 1:N
        p = softmax(v)
        logit_p = β * log(p[2] / p[1])
        h[t] = rand(BernoulliLogit(logit_p))
        i = h[t] + 1
        pred_err += reward[t, i] - v[i]
        v[i] += ForwardDiff.value(α) * pred_err 
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
N = 120
hand_p = 0.7
rh = rand(Bernoulli(hand_p), N)
lh = 1 .- rh
reward = [lh rh]


h, true_α, true_β = RWCoins(missing; N=N, reward=reward)()
# parameter recovery
model = RWCoins(h; N=N, reward=reward)
chains = sample(model, NUTS(1_000, 0.99; max_depth=20), MCMCThreads(), 3_000, 2)
chain_df = DataFrame(chains)
print("True α:", true_α, ", True β:", true_β, "\n")

density(chain_df[!, :α], label="Posterior α", xlabel="value", ylabel="Frequency", title="Posterior", color=:purple, fill=(0, 0.3))
density!(chain_df[!, :β], label="Posterior β", color=:blue, fill=(0, 0.3))
# add true values
vline!([true_α], label="True α", color=:purple, linestyle=:dash)
vline!([true_β], label="True β", color=:blue, linestyle=:dash)



