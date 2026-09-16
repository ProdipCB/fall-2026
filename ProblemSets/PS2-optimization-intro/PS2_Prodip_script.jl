#---------------------------------
# ECON 6343: Econometrics III
# Problem Set 2: script
# Author: Prodip Chakraborty
#----------------------------------


using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables

include("PS2_Prodip_source.jl")


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# The answer: argmax x* = -7.378243405540943, max f(x*) = 964.313383782421
# which matches the problem set's ~-7.38.


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# The answer: Optim, closed form (inv(X'X)X'y), and lm() all give the same
# estimates (intercept 0.6614, age -0.0046, white 0.2260, collgrad -0.0122);
# see printed output for standard errors.


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# The answer: logit estimates (intercept 0.7466, age -0.0211, white 0.9558,
# collgrad -0.0560), log-likelihood -1416.9506 at the optimum.


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# The answer: glm() reproduces the same coefficients as question 3 


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# The answer: after aggregating, occupation frequencies are 317, 264, 726, 102, 53, 246, 529 (N=2237, J=7, base = occupation 7); see printed output for the 4x6 alpha matrix. 
#Log-likelihood -3605.0904 at the optimum. Starting from zeros converges; q5_startval_check() below confirms U[0,1] and U[-1,1] starts agree to within its 1e-3 tolerance.


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 6
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# The answer: allwrap() in the source file wraps and calls every question above;
# it is called at the bottom of this script.


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 7
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# The answer: unit tests are in PS2_Prodip_tests.jl.


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# run everything (question 6)
#:::::::::::::::::::::::::::::::::::::::::::::::::::
allwrap()
println("PS2_Prodip_script.jl ran successfully.")
