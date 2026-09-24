#===========================================================
 ECON 6343: Econometrics III — Problem Set 3
 Author: Prodip Chakraborty
===========================================================#

using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

cd(@__DIR__)

# Read in the functions
include("PS3_Prodip_source.jl")

#===========================================================
 WRITEUP

 Q1. Multinomial logit
 The model has 22 parameters: 3 for each of occupations 1-7
 (occupation 8 "Other" is the base, set to 0) and one γ.
 Estimated γ̂ ≈ -0.094.

 Q2. What γ̂ means
 γ tells us how the expected wage of a job changes how much a
 person likes that job. γ̂ is small and negative: when a job's
 expected wage goes up, a woman is slightly LESS likely to pick
 it. This is surprising. One reason may be that higher-paying
 jobs have worse working conditions. Another is that Z is only
 a predicted wage, so it has error.

 Q3. Nested logit
 Nests: white collar (1-3), blue collar (4-7), other (8).
 Estimates (fill in from the output):
   λ_WC = ____   λ_BC = ____   γ = ____
 λ shows how similar the jobs inside a nest are. λ = 1 is the
 plain multinomial logit. A λ below 1 means jobs in the same
 nest are closer substitutes for each other.
===========================================================#

# Call the main function (Questions 1, 3, 4)
allwrap()
