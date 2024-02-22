using CSV, DataFrames
using Optim, Turing
using FillArrays, Distributions
using StatsFuns
using Plots, StatsPlots
using Random

function noisy_softmax(x, β)
    exp_x = exp.(β * x)
    return exp_x ./ sum(exp_x)
end

# Rescorla-Wagner model
@model function RWCoins(h, ::Type{T} = Int; N::Int) where T
    # Priors
    α ~ Normal(0, 0.3)  # learning rate
    β ~ Normal(0, 0.5)  # inverse temperature (noise/explore-exploit tradeoff)


    p₍ₜ₋₁₎ = [0.5, 0.5]
    if ismissing(h)
        h = Vector{T}(fill(undef, N))
    end
    for i in 1:N
        p = noisy_softmax(p₍ₜ₋₁₎, β)
        h[i] = rand(Categorical(p))
        δ = h[i] - p₍ₜ₋₁₎[h[i]]
        p₍ₜ₋₁₎[h[i]] += α * δ
        p₍ₜ₋₁₎[setdiff(1:2, h[i])] -= α * δ
    end
end
noisy_softmax([0.5, 0.5], 0.0)
# read simdata/W3_randomnoise.csv into a dataframe
df = DataFrame(CSV.File("simdata/W3_randomnoise.csv"))

# only include noise == 0 and rate == 0.8
df1 = filter(row -> row.noise == 0 && row.rate == 0.8, df)

# convert the dataframe to a vector
h = Vector{Int}(df1[!, :choice])
N = length(h)

# run the model
model = RWCoins(h; N=N)
chains = sample(model, NUTS(1_000, 0.99; max_depth=20), MCMCThreads(), 2_000, 2)

