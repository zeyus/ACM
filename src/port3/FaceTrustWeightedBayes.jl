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
@model function FaceTrustWeightedBayes(FirstRating::Array{Int, 2}, GroupRating::Array{Int, 2}, SecondRating::Array{Int, 2}, w1::Float64, w2::Float64, bias::Real)
    # Group level param
    μ ~ Normal(0, 1)
    σ ~ LogNormal(0, 1)

    # Individual level param
    NSubjects = size(FirstRating, 1)
    N = size(FirstRating, 2)

    # Priors
    W1 = (w1-0.5) * 2
    W2 = (w2-0.5) * 2

    # Likelihood
    for i in 1:NSubjects
        α = bias + W1 * FirstRating[i, :] + W2 * GroupRating[i, :]
        β = 2 .*
        for j in 1:N

        end
    end






    

end
