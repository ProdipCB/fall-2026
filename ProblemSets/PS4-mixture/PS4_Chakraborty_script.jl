#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 4 (Mixed Logit Models)
# Prodip Chakraborty
#---------------------------------------------------
import Pkg
haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add("ForwardDiff")

using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM,
      FreqTables, Distributions, ForwardDiff

cd(@__DIR__)
include("lgwt.jl")
include("PS4_Chakraborty_source.jl")

#---------------------------------------------------
# Question 1: Multinomial logit with Z
#---------------------------------------------------
# The code is in optimize_mlogit(). I use automatic differentiation
# (ForwardDiff) and get the standard errors from the inverse Hessian.
# I start from the PS3 Question 1 estimates.
# Result: γ̂ = 1.307 (se = 0.125).

#---------------------------------------------------
# Question 2: Does γ̂ make more sense now than in PS3?
#---------------------------------------------------
# Yes. In PS3, γ̂ was -0.094 (se 0.379). This means a higher expected
# wage makes an occupation less attractive, which does not make sense.
# Now γ̂ = 1.307 and it is positive and significant. A higher expected wage
# in an occupation makes people more likely to choose it. This is what I
# expect. The panel has many more observations (28,365), so the estimate is
# also much more precise.

#---------------------------------------------------
# Question 3: Quadrature and Monte Carlo practice
#---------------------------------------------------
# (a) With 7 points on [-4, 4], ∫φ(x)dx = 1.0045 and ∫xφ(x)dx ≈ 0.
#     So quadrature works well here.
#
# (b) Variance of N(0,2) on [-5σ, 5σ]. The true value is 4.
#       7 points:  3.266
#       10 points: 4.039
#     7 points is not good enough because the interval is wide. With 10
#     points the answer is very close to 4.
#
# (c) With D = 1,000 draws, the Monte Carlo answers are noisy and can be
#     off by a few percent. With D = 1,000,000 draws, they are very close to
#     4, 0, and 1. So Monte Carlo needs many draws to be accurate.
#
# (d) Quadrature uses a few chosen nodes with different weights. Monte Carlo
#     uses many random nodes, and every node has the same weight (b-a)/D.

#---------------------------------------------------
# Question 4: Mixed logit with quadrature
#---------------------------------------------------
# The code is in mixed_logit_quad() and optimize_mixed_logit_quad().
# I start from the Question 1 estimates and σ_γ = 1. I do not run it
# because it takes hours, so the call in allwrap() is commented out.

#---------------------------------------------------
# Question 6: Run everything
#---------------------------------------------------
allwrap()
