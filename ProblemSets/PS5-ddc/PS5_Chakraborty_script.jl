#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 5
# Prodip Chakraborty
#---------------------------------------------------

using Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

include("create_grids.jl")             # provided: state grids and transitions
include("PS5_Chakraborty_source.jl")   # all my functions

θ̂_static, result_dynamic = main()

#---------------------------------------------------
# Writeup
#---------------------------------------------------
# Question 1
#   I reshape busdataBeta0.csv to long format in load_static_data().
#   df_long has 20,000 rows (1,000 buses x 20 periods).
#
# Question 2 (static, β = 0)
#   I estimate a binary logit of Y on Odometer and Branded with GLM:
#       θ0 (constant) =  1.926  (s.e. 0.033)
#       θ1 (mileage)  = -0.148  (s.e. 0.004)
#       θ2 (brand)    =  1.059  (s.e. 0.043)
#   More mileage lowers the utility of running the bus, so replacement is
#   more likely as mileage grows. Branded buses are replaced less often.
#
# Question 3 (dynamic, β = 0.9)
#   I solve the model by backward recursion over all grid states
#   (compute_future_value!) and then compute the likelihood over the observed
#   states only (log_likelihood_dynamic). I start LBFGS at the static estimates.
#       θ0 (constant) =  1.938
#       θ1 (mileage)  = -0.151
#       θ2 (brand)    =  1.093
#       log likelihood = -8765.0
#   The signs and meaning are the same as in Question 2, and the values are
#   close.
#
# Question 4
#   Unit tests are in PS5_Chakraborty_tests.jl. It reads in the source file
#   and then tests each function.
