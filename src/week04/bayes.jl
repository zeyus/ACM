# using LazyArrays
#using DynamicHMC
using CSV, DataFrames
using Optim, Turing
using FillArrays, Distributions
using StatsFuns
using Plots, StatsPlots
using Random

# read simdata/W3_randomnoise.csv into a dataframe
df = DataFrame(CSV.File("simdata/W3_randomnoise.csv"))

# Define the Turing model
@model function simple_b_logodds(h; N::Int)
    θ ~ Normal(0, 1)
    h ~ filldist(BernoulliLogit(θ), N)
    return h
end

# only include noise == 0 and rate == 0.8
df1 = filter(row -> row.noise == 0 && row.rate == 0.8, df)

# convert the dataframe to a vector
h = df1[!, :choice]
N = length(h)

prior_draws = sample(simple_b_logodds(missing; N=N), Prior(), 2_000)

# run the model
model = simple_b_logodds(h; N=N)
chains = sample(model, NUTS(1_000, 0.99; max_depth=20), MCMCThreads(), 2_000, 2)

chain_df = DataFrame(chains)
prior_df = DataFrame(prior_draws)
# plot prior theta
density(logistic.(prior_df[!, :θ]), label="Prior θ", xlabel="θ", ylabel="Frequency", title="Prior θ", color=:blue, fill=(0, 0.3))
# plot posterior predictions
density!(logistic.(chain_df[!, :θ]), label="Posterior θ", xlabel="θ", ylabel="Frequency", title="Posterior θ", color=:purple, fill=(0, 0.3))

# add true theta of 0.8
vline!([0.8], label="True θ", color=:black, linestyle=:dash)




@model function memory_bernoulli(n::Int, h::Vector{Int}, other::Vector{Int})
    # Priors
    bias ~ Normal(0, 0.3)
    β ~ Normal(0, 0.5)

    # Initialize memory
    memory = Vector{Float64}(undef, n)
    memory[1] = 0.5

    # Update memory
    for trial in 2:n
        memory[trial] = memory[trial - 1] + (other[trial] - memory[trial - 1]) / trial
        memory[trial] = clamp(memory[trial], 0.01, 0.99)
    end

    # Likelihood
    for trial in 1:n
        h[trial] ~ BernoulliLogit(bias + β * logit(memory[trial]))
    end

    # # Generate quantities
    # bias_prior = rand(Normal(0, 0.3))
    # β_prior = rand(Normal(0, 0.5))
    # prior_preds5 = rand(Binomial(n, logistic(bias_prior + β_prior * logit(0.5))))
    # prior_preds7 = rand(Binomial(n, logistic(bias_prior + β_prior * logit(0.7))))
    # prior_preds9 = rand(Binomial(n, logistic(bias_prior + β_prior * logit(0.9))))
    # post_preds5 = rand(Binomial(n, logistic(bias + β * logit(0.5))))
    # post_preds7 = rand(Binomial(n, logistic(bias + β * logit(0.7))))
    # post_preds9 = rand(Binomial(n, logistic(bias + β * logit(0.9))))

    return bias_prior, β_prior, prior_preds5, post_preds5, prior_preds7, post_preds7, prior_preds9, post_preds9
end


@model function prior_bernoulli(n::Int, h::Vector{Int}, prior_μ::Real, prior_σ::Real)
    # Prior for θ
    θ ~ Normal(prior_μ, prior_σ)

    # Likelihood
    for trial in 1:n
        h[trial] ~ BernoulliLogit(θ)
    end

    # Generate quantities
    θ_prior = logistic(rand(Normal(0, 1)))
    θ_posterior = logistic(θ)
    prior_preds = rand(Binomial(n, θ_prior))
    posterior_preds = rand(Binomial(n, logistic(θ)))

    return θ_prior, θ_posterior, prior_preds, posterior_preds
end


@model function prior_memory(n::Int, h::Vector{Int}, prior_μ_bias::Real, prior_σ_bias::Real, prior_μ_β::Real, prior_σ_β::Real)
    # Priors for bias and β
    bias ~ Normal(prior_μ_bias, prior_σ_bias)
    β ~ Normal(prior_μ_β, prior_σ_β)
    
    # Initialize memory
    memory = Vector{Float64}(undef, n)
    memory[1] = 0.5

    # Update memory
    for trial in 2:n
        memory[trial] = memory[trial - 1] + (h[trial] - memory[trial - 1]) / trial
    end

    # Likelihood
    for trial in 1:n
        h[trial] ~ BernoulliLogit(bias + β * memory[trial])
    end

    # Generate quantities
    bias_prior = rand(Normal(prior_μ_bias, prior_σ_bias))
    β_prior = rand(Normal(prior_μ_β, prior_σ_β))
    prior_preds5 = rand(Binomial(n, logistic(bias_prior + β_prior * 0.5)))
    prior_preds7 = rand(Binomial(n, logistic(bias_prior + β_prior * 0.7)))
    prior_preds9 = rand(Binomial(n, logistic(bias_prior + β_prior * 0.9)))
    post_preds5 = rand(Binomial(n, logistic(bias + β * 0.5)))
    post_preds7 = rand(Binomial(n, logistic(bias + β * 0.7)))
    post_preds9 = rand(Binomial(n, logistic(bias + β * 0.9)))

    return bias_prior, β_prior, prior_preds5, post_preds5, prior_preds7, post_preds7, prior_preds9, post_preds9
end


@model function simple_bernoulli_logodds(h::Vector{Union{Missing, T}}, ::Type{T} = Real) where T
    # Prior for θ (on a log odds scale)
    θ ~ Normal(0, 1)
    
    # Likelihood (Bernoulli distribution with logit link)
    # DynamicPPL.@addlogprob! sum(logpdf.(BernoulliLogit.(θ), h))
    n = length(h)
    for trial in 1:n
        h[trial] ~ BernoulliLogit(θ)
    end

    # Generate quantities
    θ_prior = logistic(rand(Normal(0, 1)))
    θ_posterior = logistic(θ)
    prior_preds ~ Binomial(n, θ_prior)
    posterior_preds ~ Binomial(n, θ_posterior)

    # return θ_prior, θ_posterior, prior_preds, posterior_preds
end

# Example usage
n = 100  # Number of trials
h = Vector{Union{Missing,Int}}(undef, n)
model = simple_bernoulli_logodds(h, Int)
# dynamic_nuts = externalsampler(DynamicHMC.NUTS())
chains = sample(model, Turing.NUTS(1000, 0.99), MCMCThreads(), 2_000, 4)

# Plot the results
plot(chains)
chains