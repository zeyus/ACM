"""A turing Model to use the data from
Socially Learned Attitude Change is not reduced in Medicated Patients with Schizophrenia
Arndis Simonsen, Riccardo Fusaroli, Joshua Charles Skewes, Andreas Roepstorff, Ole Mors, Vibeke Bliksted, Daniel Campbell-Meiklejohn

Data structure:
- ID: an identifier of the participant
- FaceID: an identifier of the specific face rated
- FirstRating: the trustworthiness rating (1-8) given by the participant BEFORE seeing other ratings   
- GroupRating: the trustworthiness rating (1-8) given by others
- SecondRating: the trustworthiness rating (1-8) given after seeing the others (at second exposure)
"""

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
using ParetoSmooth

include("../LogCommon.jl")

@info "Preparing model..."

# Simple (non-weighted) bayesian model for the trustworthiness data
@model function FaceTrustSimpleBayes(
    FirstRating::Union{Array{Int, 2}, Missing},
    GroupRating::Union{Array{Int, 2}, Missing},
    SecondRating::Union{Array{Int, 2}, Missing},
    N::Int = 10,
    Ntrials::Int = 10,
    lb::Int = 0,
    ub::Int = 7)

    # create rating arrays if missing
    if isa(FirstRating, Missing)
        FirstRating = rand(lb:ub, N, Ntrials)
    end
    if isa(GroupRating, Missing)
        GroupRating = rand(lb:ub, N, Ntrials)
    end

    # Group level param
    μbias ~ Normal(0, 1)
    σbias ~ Normal(0, 0.2)

    # Subjec level param
    subject_bias ~ filldist(Normal(0, 1), N)
    bias = logistic.(μbias .+ σbias .* subject_bias)
    
    log_lik = zeros(N, Ntrials)

    cats = ub-lb

    β = 2 * cats # e.g 2 * 7 = 14

    for i in 1:N
        α = FirstRating[i, :] .+ GroupRating[i, :]  # e.g 2 + 3 = 5
        
        for j in 1:Ntrials
            SecondRating[i, j] ~ BetaBinomial(cats, 1+α[j]+bias[i], 1+(β-α[j])*bias[i])  # e.g 1+5+0.5, 1+9*0.5
            log_lik[i, j] = logpdf(BetaBinomial(cats, 1+ReverseDiff.value(α[j])+ReverseDiff.value(bias[i]), 1+(β-ReverseDiff.value(α[j]))*ReverseDiff.value(bias[i])), ReverseDiff.value(SecondRating[i, j]))
        end
    end
    return (SecondRating, log_lik, mean(log_lik), sum(log_lik))
end


# Weighted bayes model for the trustworthiness data
# we will atempt to predict the second rating given the first rating and the group rating
# Outcome ≈ bias + W1 * Source1 +W2  * Source2
# Rating arrays are of size NSubjects x N
@model function FaceTrustWeightedBayes(
    FirstRating::Union{Array{Int, 2}, Missing},
    GroupRating::Union{Array{Int, 2}, Missing},
    SecondRating::Union{Array{Int, 2}, Missing},
    N::Int = 10,
    Ntrials::Int = 10,
    lb::Int = 0,
    ub::Int = 7)
    # Group level param
    μbias ~ Normal(0, 1)
    σbias ~ Normal(0, 0.2)

    # Subjec level param
    subject_bias ~ filldist(Normal(0, 1), N)
    W1 = logistic.(subject_bias .+ μbias .+ σbias .* subject_bias)*2
    W2 = 2 .- W1

    log_lik = zeros(N, Ntrials)

    β = 14
    cats = ub-lb
    for i in 1:N
        α = W1[i] * FirstRating[i, :] .+ W2[i] .* (GroupRating[i, :])
        for j in 1:Ntrials
            SecondRating[i, j] ~ BetaBinomial(cats, 1+α[j], 1+β-α[j])
            log_lik[i, j] = logpdf(BetaBinomial(cats, 1+α[j], 1+β-α[j]), SecondRating[i, j])
        end
    end
    return (SecondRating, log_lik, mean(log_lik), sum(log_lik))
end

@info "Setting default sampling params..."
n_samples = 1_000 # cmdstan default
ntune = 1_000 # cmdstan default
nchains = 2 # cmdstan default is 1 but let's use 2
δ = 0.8 # cmdstan default
max_depth = 10 # cmdstan default

@info "Generating data..."

μbiases = [0.25, 0.5, 0.75]

sim_results = Dict()
for μbias in μbiases
    @info "Generating data for μbias=$μbias..."
    NSubj = 40
    NTrials = 153
    FirstRating = zeros(Int, NSubj, NTrials)
    GroupRating = zeros(Int, NSubj, NTrials)
    SecondRating = zeros(Int, NSubj, NTrials)
    @info "Generating ratings data..."
    subject_biases = zeros(NSubj)
    for i in 1:NSubj 
        subject_diff = rand(Normal(0, 0.01))
        bias = μbias + subject_diff
        bias = clamp(bias, 0.01, 0.99)
        subject_biases[i] = bias
        FirstRating[i, :] = round.(Int, rand(Uniform(0, 7), NTrials))
        GroupRating[i, :] = round.(Int, rand(Uniform(0, 7), NTrials))

        # second rating
        α = FirstRating[i, :] .+ GroupRating[i, :]
        β = 2 * 7
        # sample from beta distribution
        for j in 1:NTrials
            belief = rand(Beta(1+α[j]*bias, 1+(β-α[j])*bias))
            SecondRating[i, j] = rand(Binomial(7, belief))
        end
    end
    # fit model
    @info "Running model...simplebayes"
    modelSB = FaceTrustSimpleBayes(FirstRating, GroupRating, SecondRating, NSubj, NTrials)
    chainSB = sample(modelSB, NUTS(ntune, δ, max_depth=max_depth; adtype=Turing.AutoReverseDiff(true)), MCMCThreads(), n_samples, nchains, progress=true, verbose=true)
    @info "Saving results to sim_results dict..."
    sim_results["simplebayes_μbias_$μbias"] = Dict(
        "chains" => chainSB,
        "FirstRating" => FirstRating,
        "GroupRating" => GroupRating,
        "SecondRating" => SecondRating,
        "μbias" => μbias,
        "model" => modelSB,
        "subject_biases" => subject_biases,
        "generated_quantities" => generated_quantities(modelSB, chainSB)
    )
end

@info "Loading data..."
data = CSV.File("./data/Simonsen_clean.csv") |> DataFrame

NSubj = length(unique(data.ID))
NTrials = 153
# convert ratings to matrices by subject
Tdata = groupby(data, :ID)
IDs = zeros(Int, NSubj)
FirstRating = zeros(Int, NSubj, NTrials)
GroupRating = zeros(Int, NSubj, NTrials)
SecondRating = zeros(Int, NSubj, NTrials)
for (i, subdf) in enumerate(Tdata)
    IDs[i] = subdf.ID[1]
    # remove one to make it 0-indexed
    FirstRating[i, :] = subdf.FirstRating .- 1
    GroupRating[i, :] = subdf.GroupRating .- 1
    SecondRating[i, :] = subdf.SecondRating .- 1
end
# run the model
@info "Running model...simplebayes"
modelSB = FaceTrustSimpleBayes(FirstRating, GroupRating, SecondRating, NSubj, NTrials)
# chain = sample(model, NUTS(1_500, 0.65; adtype=Turing.AutoForwardDiff()), MCMCThreads(), 2_000, 4, progress=true, verbose=true)
chainSB = sample(modelSB, NUTS(1_500, 0.65; adtype=Turing.AutoReverseDiff(true)), MCMCThreads(), 2_000, 4, progress=true, verbose=true)


# save chain
@info "Saving chain..."
h5open("chains/fitted_simplebayes_facetrust.h5", "cw") do file
    g = create_group(file, "simplebayes")
    write(g, chainSB)
end

# show fitted vs true values
@info "Plotting fitted vs true values..."
μsamples = chainSB[:μbias][:, 1]
σsamples = chainSB[:σbias][:, 1]

predicted_ratings = zeros(Int, NSubj, NTrials, 2_000)
mode_predicted_ratings = zeros(Int, NSubj, NTrials)
β = 2 * 7
for (idx, (μ, σ)) in enumerate(zip(μsamples, σsamples))
    for i in 1:NSubj
        α = FirstRating[i, :] .+ GroupRating[i, :]
        bias = logistic.(μ + σ * chainSB["subject_bias[$i]"][:, 1][idx])
        for j in 1:NTrials
            predicted_ratings[i, j, idx] = rand(BetaBinomial(7, 1+α[j]+bias, 1+(β-α[j])*bias))+1
        end
    end
end
mode_predicted_ratings = mapslices(mode, predicted_ratings, dims=3)[:, :, 1]
# plot predicted vs true using mode
p = scatter(mode_predicted_ratings[1, :], label="Predicted", xlabel="Trial", ylabel="Rating", title="Predicted vs True Ratings", dpi=300)
scatter!(FirstRating[1, :].+1, label="True", color=:red)

# plot error by trial
error = mode_predicted_ratings .- (SecondRating .+ 1)
p2 = plot(error[3, :], label="Error", xlabel="Trial", ylabel="Error", title="Mean Error by Trial", dpi=300)






modelWB = FaceTrustWeightedBayes(FirstRating, GroupRating, SecondRating, NSubj, NTrials)
chainWB = sample(modelWB, NUTS(1_500, 0.65; adtype=Turing.AutoReverseDiff(true)), MCMCThreads(), 2_000, 4, progress=true, verbose=true)

# save chain
@info "Saving chain..."
h5open("chains/fitted_weightedbayes_facetrust.h5", "cw") do file
    g = create_group(file, "weightedbayes")
    write(g, chainWB)
end

# show fitted vs true values
@info "Plotting fitted vs true values..."
μsamples = chainWB[:μbias][:, 1]
σsamples = chainWB[:σbias][:, 1]

predicted_ratings = zeros(Int, NSubj, NTrials, 2_000)
mode_predicted_ratings = zeros(Int, NSubj, NTrials)
β = 2 * 7

for (idx, (μ, σ)) in enumerate(zip(μsamples, σsamples))
    for i in 1:NSubj
        α = FirstRating[i, :] .+ GroupRating[i, :]
        W1 = logistic.(chainWB["subject_bias[$i]"][:, 1][idx] .+ μ + σ * chainWB["subject_bias[$i]"][:, 1][idx])*2
        W2 = 2 .- W1
        for j in 1:NTrials
            predicted_ratings[i, j, idx] = rand(BetaBinomial(7, 1+W1*α[j], 1+W2*(β-α[j])))+1
        end
    end
end

mode_predicted_ratings = mapslices(mode, predicted_ratings, dims=3)[:, :, 1]

# plot predicted vs true using mode
p3 = scatter(mode_predicted_ratings[1, :], label="Predicted", xlabel="Trial", ylabel="Rating", title="Predicted vs True Ratings", dpi=300)
scatter!(FirstRating[1, :].+1, label="True", color=:red)

# plot error by trial
error = mode_predicted_ratings .- (SecondRating)
p4 = plot(error[5, :], label="Error", xlabel="Trial", ylabel="Error", title="Mean Error by Trial", dpi=300)


# testing ParetoSmooth pkg
@info "Testing ParetoSmooth..."
psis(modelWB, chainWB)
