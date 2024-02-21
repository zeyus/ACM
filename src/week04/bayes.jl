using DynamicHMC: NUTS
using Turing
using Optim
using Distributions
using StatsFuns
using Plots, StatsPlots
using DataFrames
using FillArrays

# Define the Turing model
@model function example_1(n::Int, h::Vector{Int})
    # Prior for θ (log odds scale)
    θ ~ Normal(0, 1)
    
    # Likelihood: Bernoulli distribution with logit link
    for i in 1:n
        h[i] ~ BernoulliLogit(θ)
    end
    
    # Generate quantities
    θ_prior = logistic(rand(Normal(0, 1)))
    θ_posterior = logistic(θ)
    prior_preds = rand(Binomial(n, θ_prior))
    posterior_preds = rand(Binomial(n, logistic(θ)))
    
    return θ_prior, θ_posterior, prior_preds, posterior_preds
end


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

    # Generate quantities
    bias_prior = rand(Normal(0, 0.3))
    β_prior = rand(Normal(0, 0.5))
    prior_preds5 = rand(Binomial(n, logistic(bias_prior + β_prior * logit(0.5))))
    prior_preds7 = rand(Binomial(n, logistic(bias_prior + β_prior * logit(0.7))))
    prior_preds9 = rand(Binomial(n, logistic(bias_prior + β_prior * logit(0.9))))
    post_preds5 = rand(Binomial(n, logistic(bias + β * logit(0.5))))
    post_preds7 = rand(Binomial(n, logistic(bias + β * logit(0.7))))
    post_preds9 = rand(Binomial(n, logistic(bias + β * logit(0.9))))

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
    h ~ filldist(BernoulliLogit(θ), size(h, 2))
    # for trial in 1:length(h)
    #     h[trial] ~ BernoulliLogit(θ)
    # end

    # Generate quantities
    θ_prior = logistic(rand(Normal(0, 1)))
    θ_posterior = logistic(θ)
    prior_preds ~ Binomial(n, θ_prior)
    posterior_preds ~ Binomial(n, logistic(θ))

    return θ_prior, θ_posterior, prior_preds, posterior_preds
end

# Example usage
n = 100  # Number of trials
h = Vector{Union{Missing,Int}}(undef, n)
model = simple_bernoulli_logodds(h, Int)
chains = sample(model, NUTS(1000, 0.99), MCMCThreads(), 2000, 2)

# Plot the results
plot(chains)
plot_sampler(chains)