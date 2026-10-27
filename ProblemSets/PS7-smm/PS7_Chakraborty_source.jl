################################################################################
# Problem Set 7 - Source file (all functions)
# ECON 6343: Econometrics III
# Prodip Chakraborty
################################################################################

using Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames,
      DataFramesMeta, CSV, HTTP, GLM

################################################################################
# Data loading
################################################################################

"""
    load_data(url)

Load the nlsw88 data. Build the wage regression from PS2 Q2:
    log(wage) = β0 + β1 age + β2 white + β3 collgrad + u
Returns (df, X, y).
"""
function load_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(df, 1)) df.age (df.race .== 1) (df.collgrad .== 1)]
    X = Float64.(X)
    y = log.(df.wage)
    return df, X, y
end

"""
    prepare_occupation_data(df)

Data for the multinomial logit from PS2 Q5. Drops missing occupation and
collapses categories 8-13 into category 7, so J = 7.
Returns (df, X, y).
"""
function prepare_occupation_data(df)
    df = dropmissing(df, :occupation)
    df[df.occupation .>= 8, :occupation] .= 7
    df.white = df.race .== 1
    X = Float64.([ones(size(df, 1)) df.age df.white df.collgrad])
    y = Int.(df.occupation)
    return df, X, y
end

################################################################################
# Helpers for the multinomial logit
################################################################################

"""
    choice_dummies(y, J)

N×J matrix d with d[i,j] = 1 if y[i] == j.
"""
function choice_dummies(y, J)
    N = length(y)
    d = zeros(N, J)
    for j in 1:J
        d[:, j] = y .== j
    end
    return d
end

"""
    mlogit_probs(α, X, J)

Choice probabilities P (N×J). α is the K(J-1) vector of coefficients;
the coefficients of alternative J are normalized to 0.
Uses the max-subtraction trick so exp() does not overflow.
"""
function mlogit_probs(α, X, J)
    K = size(X, 2)
    bigα = [reshape(α, K, J - 1) zeros(K)]
    v = X * bigα
    v = v .- maximum(v, dims = 2)
    num = exp.(v)
    return num ./ sum(num, dims = 2)
end

################################################################################
# Question 1: OLS via GMM
################################################################################

"""
    ols_gmm(β, X, y)

GMM objective for OLS (slide #8).
Moment function (one moment per observation):  g_i(β) = y_i - X_i β,
so g = y - Xβ is N×1. Objective: J(β) = g' W g.
Default W = I, the N×N identity matrix. (Julia's `I` acts as an identity
of whatever size is needed, here N×N, without storing N² numbers.
Passing W = Matrix(I, N, N) gives the same answer.)
"""
function ols_gmm(β, X, y; W = I)
    ŷ = X * β
    g = y .- ŷ
    J = g' * W * g
    return J
end

"""
    ols_closed_form(X, y)

Closed-form OLS estimator (X'X)^(-1) X'y.
"""
ols_closed_form(X, y) = inv(X' * X) * (X' * y)

################################################################################
# Question 2: Multinomial logit by MLE and GMM
################################################################################

"""
    mlogit_mle(α, X, y)

Negative log likelihood of the multinomial logit (PS2 Q5).
"""
function mlogit_mle(α, X, y)
    J = length(unique(y))
    d = choice_dummies(y, J)
    P = mlogit_probs(α, X, J)
    loglike = -sum(d .* log.(P))
    return loglike
end

"""
    mlogit_gmm_overid(α, X, y)

GMM objective for Q2(b)-(c). g = d - P, where d and P are stacked into
N·J vectors (length(g) == N*J). Objective g'Wg with W = I (NJ×NJ).
"""
function mlogit_gmm_overid(α, X, y; W = I)
    J = length(unique(y))
    d = choice_dummies(y, J)
    P = mlogit_probs(α, X, J)
    g = d[:] .- P[:]
    return g' * W * g
end

"""
    mlogit_gmm(α, X, y)

Extra (not required): just-identified GMM using the score moments
E[X_i (d_ij - P_ij)] = 0, j = 1..J-1. These are the MLE first-order
conditions, so the solution equals the MLE.
"""
function mlogit_gmm(α, X, y)
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    d = choice_dummies(y, J)
    P = mlogit_probs(α, X, J)
    g = vec(X' * (d[:, 1:J-1] .- P[:, 1:J-1])) ./ N   # K(J-1) moments
    return N * dot(g, g)
end

"""
    mlogit_starting_values(df, X; J = 7)

Starting values for the MLE: one linear probability model per occupation,
each differenced from the base category J (same idea as the starter code).
"""
function mlogit_starting_values(df, X; J = 7)
    K = size(X, 2)
    svals = zeros(K, J)
    for j in 1:J
        dj = Float64.(df.occupation .== j)
        svals[:, j] = X \ dj
    end
    svals = svals[:, 1:J-1] .- svals[:, J]
    return svals[:]
end

################################################################################
# Question 3: Simulate multinomial logit data
################################################################################

"""
    sim_logit(N, J, β; seed = nothing)

Simulate multinomial logit data with the method in Q3 (a)-(e).
- β is K×J. Column J should be zeros (normalization).
- X: intercept plus K-1 normal covariates.
- P: N×J choice probabilities.
- ε: N×1 U[0,1] draws.
- Y_i = Σ_j 1{ Σ_{k=j}^J P_ik > ε_i }.
Returns (X, Y).
"""
function sim_logit(N, J, β; seed = nothing)
    seed === nothing || Random.seed!(seed)
    K = size(β, 1)
    @assert size(β, 2) == J "β must be K×J"
    @assert J > 2 && K > 1 "need J > 2 and K > 1"

    # (a) X: intercept + K-1 covariates
    X = hcat(ones(N), randn(N, K - 1))

    # (c) choice probabilities
    v = X * β
    v = v .- maximum(v, dims = 2)
    P = exp.(v) ./ sum(exp.(v), dims = 2)

    # (d) preference shocks
    ε = rand(N)

    # (e) choices
    Y = zeros(Int, N)
    for j in 1:J
        Y .+= vec(sum(P[:, j:J], dims = 2) .> ε)
    end
    return X, Y
end

"""
    gumbel_choices(X, β, ε)

Q3(f) choice rule: Y_i = argmax_j X_i β_j + ε_ij, with ε an N×J matrix of
T1EV (Gumbel) draws. Used by sim_logit_w_gumbel (Q3) AND by the SMM
objective (Q5), so Q5 is built on the Q3 code.
"""
gumbel_choices(X, β, ε) = argmax.(eachrow(X * β .+ ε))

"""
    sim_logit_w_gumbel(N, J, β; seed = nothing)

Q3(f) alternative: draw an N×J matrix of T1EV (Gumbel) shocks and set
Y_i = argmax_j X_i β_j + ε_ij. Returns (X, Y).
"""
function sim_logit_w_gumbel(N, J, β; seed = nothing)
    seed === nothing || Random.seed!(seed)
    K = size(β, 1)
    X = hcat(ones(N), randn(N, K - 1))
    ε = rand(Gumbel(0, 1), N, J)
    Y = gumbel_choices(X, β, ε)
    return X, Y
end

"""
    default_beta()

β used for the Q3 simulation: K = 3 covariates, J = 4 alternatives.
"""
default_beta() = hcat([1.0, -1.0, 0.5],
                      [-2.0, 0.5, 0.3],
                      [0.0, -0.5, 2.0],
                      zeros(3))

################################################################################
# Question 5: Multinomial logit by SMM
################################################################################

"""
    draw_gumbel_shocks(N, J, D; seed = 1234)

Draw the D sets of N×J Gumbel shocks ONCE (common random numbers).
Keeping them fixed across objective evaluations is what makes the SMM
objective a fixed function of α.
"""
function draw_gumbel_shocks(N, J, D; seed = 1234)
    rng = MersenneTwister(seed)
    return [rand(rng, Gumbel(0, 1), N, J) for _ in 1:D]
end

"""
    mlogit_smm_overid(α, X, y, shocks)

SMM objective (slide #18 idea). For each draw d, simulate choices with the
Q3(f) code, gumbel_choices(X, bigα, ε):  ỹ_i = argmax_j X_i α_j + ε_ij.
The simulated choice frequencies bigỸ (N×J) replace P in the GMM moments:
g = d - bigỸ stacked as an N·J vector, W = I, J(α) = g'g.
"""
function mlogit_smm_overid(α, X, y, shocks)
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    D = length(shocks)
    bigY = choice_dummies(y, J)
    bigα = [reshape(α, K, J - 1) zeros(K)]
    bigỸ = zeros(N, J)
    for ε in shocks
        ỹ = gumbel_choices(X, bigα, ε)          # Q3 code
        for j in 1:J
            bigỸ[:, j] .+= (ỹ .== j) ./ D
        end
    end
    g = bigY[:] .- bigỸ[:]
    return g' * I * g
end

################################################################################
# Output helper
################################################################################

"""
    coef_table(α, K, J; names = ["const", "age", "white", "collgrad"])

Return the K×(J-1) coefficient matrix as a string table (one row per covariate).
"""
function coef_table(α, K, J; names = ["const", "age", "white", "collgrad"])
    A = reshape(α, K, J - 1)
    rows = ["  " * rpad(names[k], 9) * join(lpad.(string.(round.(A[k, :], digits = 3)), 9))
            for k in 1:K]
    return join(rows, "\n")
end
