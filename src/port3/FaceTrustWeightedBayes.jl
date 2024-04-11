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
include("../LogCommon.jl")

@info "Preparing model..."

# Weighted bayes model for the trustworthiness data
# we will atempt to predict the second rating given the first rating and the group rating
# Outcome ≈ bias + W1 * Source1 +W2  * Source2
# Rating arrays are of size NSubjects x N
@model function FaceTrustWeightedBayes(FirstRating::Array{Int, 2}, GroupRating::Array{Int, 2}, SecondRating::Array{Int, 2})
    # Group level param
    μ ~ Normal(0, 1)
    σ ~ Normal(0, 0.2)
    # Individual level param
    N = size(FirstRating, 1)
    NRatings = size(FirstRating, 2)
    logweight1 ~ filldist(Normal(0, 1), N)

    W1 = logistic.(logweight1 .+ (μ .+ σ .* logweight1))
    W2 = 1 .- W1

    β = repeat([2*7], NRatings)
    for i in 1:N
        α = W1[i] * FirstRating[i, :] .+ W2[i] .* (GroupRating[i, :] .- 2)
        for j in 1:NRatings
            SecondRating[i, j] ~ BetaBinomial(NRatings, 1+α[j], 1+β[j]-α[j])
        end
    end
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
    FirstRating[i, :] = subdf.FirstRating
    GroupRating[i, :] = subdf.GroupRating
    SecondRating[i, :] = subdf.SecondRating
end
# run the model
@info "Running model..."
model = FaceTrustWeightedBayes(FirstRating, GroupRating, SecondRating)
chain = sample(model, NUTS(1_500, 0.65; adtype=Turing.AutoReverseDiff(true)), MCMCThreads(), 2_000, 4, progress=true, verbose=true)
