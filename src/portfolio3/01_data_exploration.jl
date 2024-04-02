# This file creates some summary statistics / explorational plots
# data from https://pubmed.ncbi.nlm.nih.gov/30700729/
# Socially Learned Attitude Change is not reduced in Medicated Patients with Schizophrenia
# Simonsen, Arndis; Fusaroli, Riccardo; Skewes, Joshua; Roepstorff, Andreas; Mors, Ole; Campbell-Meiklejohn, Daniel
# - ID: an identifier of the participant
# - FaceID: an identifier of the specific face rated
# - FirstRating: the trustworthiness rating (1-8) given by the participant BEFORE seeing other ratings   
# - GroupRating: the trustworthiness rating (1-8) given by others
# - SecondRating: the trustworthiness rating (1-8) given after seeing the others (at second exposure)
using CSV, DataFrames
using Statistics
using StatsFuns
using Plots, StatsPlots
using CategoricalArrays

# Load the data
# file 1: Simonsen_clean.csv: trustworthiness ratings of faces from patients with schizophrenia
patient_data = CSV.read("data/Simonsen_clean.csv", DataFrame)

# file 2: cogsci_clean.csv: trustworthiness ratings of faces from cogsci students (pre pandemic)
control_data = CSV.read("data/cogsci_clean.csv", DataFrame)

# file 3: sc_df_clean.csv: trustworthiness ratings of faces from cogsci students (peri pandemic)
control_data2 = CSV.read("data/sc_df_clean.csv", DataFrame)

# create a dictionary of the datasets
data_dict = Dict("patient" => patient_data, "cs_pre" => control_data, "cs_peri" => control_data2)

# create a dataframe for the summary statistics
summary_df = DataFrame(
    dataset = String[],
    measure = String[],
    mean = Float64[],
    median = Float64[],
    std = Float64[]
)

# loop through the datasets to get the mean, median, and std of the ratings
for (key, value) in data_dict
    # get the mean, median, and std of the first ratings
    push!(summary_df, [key, "FirstRating", mean(value.FirstRating), median(value.FirstRating), std(value.FirstRating)])
    # get the mean, median, and std of the second ratings
    push!(summary_df, [key, "SecondRating", mean(value.SecondRating), median(value.SecondRating), std(value.SecondRating)])
    # get the mean, median, and std of the change in ratings
    push!(summary_df, [key, "Change", mean(value.Change), median(value.Change), std(value.Change)])
    # get the mean, median, and std of the group ratings
    push!(summary_df, [key, "GroupRating", mean(value.GroupRating), median(value.GroupRating), std(value.GroupRating)])
end
  
summary_df
# plot the summary statistics
@df summary_df groupedbar(:measure, :mean, group=:dataset, yerr=:std, title="Trustworthiness rating data summary", ylabel="Mean", xlabel="Dataset", legend=:topleft)

# plot the distribution of the first ratings
@df patient_data histogram(:FirstRating, title="Distribution of First Ratings", xlabel="Rating", ylabel="Frequency", label="Patient Data", alpha=0.6)
@df control_data histogram!(:FirstRating, title="Distribution of First Ratings", xlabel="Rating", ylabel="Frequency", label="Control Data (Pre Pandemic)", alpha=0.6)
@df control_data2 histogram!(:FirstRating, title="Distribution of First Ratings", xlabel="Rating", ylabel="Frequency", label="Control Data (Peri Pandemic)", alpha=0.6)

# add a dataset column to the dataframes
patient_data.dataset .= "Patient"
control_data.dataset .= "Control (Pre Pandemic)"
control_data2.dataset .= "Control (Peri Pandemic)"

# replace ID with Participant in the control_data2 dataframe
# drop the ID col
select!(control_data2, Not(:ID))
# rename the Participant col
rename!(control_data2, :Participant => :ID)
# now drop TimeStamp1 and TimeStamp2
select!(control_data2, Not(:TimeStamp1))
select!(control_data2, Not(:TimeStamp2))
# combine the dataframes
combined_data = vcat(patient_data, control_data, control_data2)

# find faceIDs that exist in all datasets
faceIDs = intersect(intersect(unique(patient_data.FaceID), unique(control_data.FaceID)), unique(control_data2.FaceID))
# plot the mean first ratings for 10 random faceIDs
random_faceIDs = rand(faceIDs, 10)
faceID_data = filter(row -> row.FaceID in random_faceIDs, combined_data)
# make faceID categorical
faceID_data.FaceID = categorical(faceID_data.FaceID)
@df faceID_data groupedbar(string.(:FaceID), :FirstRating, group=:dataset, title="Mean First Ratings for Random FaceIDs", xlabel="FaceID", ylabel="Mean First Rating", legend=:topleft)