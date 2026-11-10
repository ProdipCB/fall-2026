#==================================================================================
Problem Set 8 - Source Code (functions only)
ECON 6343: Econometrics III
==================================================================================#

using Random, LinearAlgebra, Statistics, Distributions
using Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM
using MultivariateStats, ForwardDiff

# Names of the six ASVAB test scores (always select by name, never by position)
const ASVAB_COLS = [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]

# Automatic-differentiation backend for Optim. Newer Optim versions use ADTypes;
# older versions use the symbol :forward. This line works with both.
const AD_BACKEND = isdefined(Optim, :ADTypes) ? Optim.ADTypes.AutoForwardDiff() : :forward

#==================================================================================
# Gauss-Legendre quadrature (lgwt.jl from Problem Set 4)
==================================================================================#

"""
    lgwt(N0, a=-1, b=1)

Legendre-Gauss nodes `x` and weights `w` on [a,b] with N0 points.
∫ f(x) dx over [a,b] ≈ sum(f.(x) .* w).
Written by Greg von Winckel (2004); adapted to Julia by Tyler Ransom.
"""
function lgwt(N0::Integer, a::Real=-1, b::Real=1)
    N  = N0 - 1
    N1 = N + 1
    N2 = N + 2

    xu = range(-1, stop=1, length=N1)

    # Initial guess
    y = cos.((2*(0:N) .+ 1)*pi/(2*N .+ 2)) .+ (0.27/N1) .* sin.(pi .* xu .* N/N2)

    # Legendre-Gauss Vandermonde Matrix and its derivative
    L  = zeros(N1, N2)
    Lp = zeros(N1, N2)

    # Newton-Raphson on the zeros of the (N+1)th Legendre polynomial
    y0 = 2
    vareps = 2e-52
    i = 0
    tracker = 0
    it_max = 10
    while (norm(y .- y0, Inf) > vareps && tracker <= it_max)
        d = norm(y .- y0, Inf)
        L[:, 1] .= 1
        Lp[:, 1] .= 0
        L[:, 2] .= y
        for k = 2:N1
            L[:, k+1] = ((2*k-1)*y .* L[:, k] .- (k-1)*L[:, k-1]) / k
        end
        Lp = (N2)*(L[:, N1] .- y .* L[:, N2]) ./ (1 .- y.^2)
        y0 = y
        y  = y0 - L[:, N2] ./ Lp
        if norm(y .- y0, Inf) == d
            tracker += 1
        end
        i += 1
    end

    # Linear map from [-1,1] to [a,b]
    x = (a .* (1 .- y) .+ b .* (1 .+ y)) ./ 2

    # Weights
    w = (b - a) ./ ((1 .- y.^2) .* Lp.^2) * (N2/N1)^2

    return x, w
end

#==================================================================================
# Question 1: Load data and estimate base regression
==================================================================================#

"""
    load_data(src::String)

Load the NLSY data. `src` can be a local file path or a URL.
"""
function load_data(src::String)
    if isfile(src)
        return CSV.read(src, DataFrame)
    else
        return CSV.read(HTTP.get(src).body, DataFrame)
    end
end

"""
    estimate_base_regression(df)

OLS: logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr
"""
function estimate_base_regression(df::DataFrame)
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)
end

#==================================================================================
# Question 2: Correlations among ASVAB scores
==================================================================================#

"""
    compute_asvab_correlations(df)

Return the 6×6 correlation matrix of the ASVAB scores as a DataFrame
(column names = ASVAB test names; rows are in the same order).
"""
function compute_asvab_correlations(df::DataFrame)
    asvabMat = Matrix{Float64}(df[:, ASVAB_COLS])
    C = cor(asvabMat)
    return DataFrame(C, string.(ASVAB_COLS))
end

#==================================================================================
# Question 3: Regression with all six ASVAB scores
==================================================================================#

"""
    estimate_full_regression(df)

OLS of logwage on the Q1 covariates plus all six ASVAB scores.
"""
function estimate_full_regression(df::DataFrame)
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr +
                       asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), df)
end

#==================================================================================
# Question 4: PCA regression
==================================================================================#

"""
    asvab_index(df, method)

Fit `method` (PCA or FactorAnalysis) with one dimension on the six ASVAB scores
and return the N-vector of scores. MultivariateStats wants a J×N matrix, so the
data are transposed. The sign of a component/factor is arbitrary, so I flip it
(if needed) so that a higher score means higher average ASVAB.
"""
function asvab_index(df::DataFrame, method)
    asvabMat = Matrix{Float64}(df[:, ASVAB_COLS])'          # J×N
    M = fit(method, asvabMat; maxoutdim=1)
    scores = vec(MultivariateStats.transform(M, asvabMat))  # 1×N -> N-vector
    avg = vec(mean(asvabMat; dims=1))
    if cor(scores, avg) < 0
        scores = -scores
    end
    return scores
end

"""
    estimate_pca_regression(df)

Add the first principal component of the ASVABs to the Q1 regression.
"""
function estimate_pca_regression(df::DataFrame)
    df2 = copy(df)                         # do not change the caller's DataFrame
    df2.asvabPCA = asvab_index(df, PCA)
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabPCA), df2)
end

#==================================================================================
# Question 5: Factor Analysis regression
==================================================================================#

"""
    estimate_factor_regression(df)

Add the first factor (FactorAnalysis) of the ASVABs to the Q1 regression.
"""
function estimate_factor_regression(df::DataFrame)
    df2 = copy(df)
    df2.asvabFactor = asvab_index(df, FactorAnalysis)
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabFactor), df2)
end

#==================================================================================
# Question 6: Full factor model by maximum likelihood
==================================================================================#

"""
    prepare_factor_matrices(df)

Returns
- `X`      : N×7 wage covariates (black, hispanic, female, schoolt, gradHS, grad4yr, constant)
- `y`      : N log wages
- `Xfac`   : N×4 measurement covariates (black, hispanic, female, constant)
- `asvabs` : N×6 ASVAB scores
"""
function prepare_factor_matrices(df::DataFrame)
    N = nrow(df)
    X = hcat(Float64.(df.black), Float64.(df.hispanic), Float64.(df.female),
             Float64.(df.schoolt), Float64.(df.gradHS), Float64.(df.grad4yr), ones(N))
    y = Float64.(df.logwage)
    Xfac = hcat(Float64.(df.black), Float64.(df.hispanic), Float64.(df.female), ones(N))
    asvabs = Matrix{Float64}(df[:, ASVAB_COLS])
    return X, y, Xfac, asvabs
end

"""
    unpack_params(θ, K, L, J)

Split θ into (α, β, γ, δ, σ). Order inside θ:
  α : L×J measurement coefficients (stored column by column, θ[1:J*L])
  β : K wage coefficients
  γ : J measurement factor loadings
  δ : 1 wage factor loading (scalar)
  σ : J+1 standard deviations (J measurements, then wage)
Total length = J*L + K + 2J + 2.
"""
function unpack_params(θ::AbstractVector, K::Integer, L::Integer, J::Integer)
    @assert length(θ) == J*L + K + 2J + 2 "θ has the wrong length"
    α = reshape(θ[1:J*L], L, J)
    β = θ[J*L+1 : J*L+K]
    γ = θ[J*L+K+1 : J*L+K+J]
    δ = θ[J*L+K+J+1]
    σ = θ[J*L+K+J+2 : J*L+K+2J+2]
    return α, β, γ, δ, σ
end

"""
    factor_model(θ, X, Xfac, Meas, y, R)

Negative log likelihood of the measurement system

    asvab_j = Xfac*α_j + γ_j*ξ + ε_j,   ε_j ~ N(0, σ_j²),  j = 1..J
    logwage = X*β     + δ*ξ   + ε,     ε   ~ N(0, σ_w²)
    ξ ~ N(0,1)

ξ is integrated out with R-point Gauss-Legendre quadrature on [-5,5]:

    L_i = Σ_r ω_r [∏_j φ((M_ij - Xfac_i α_j - γ_j ξ_r)/σ_j)/σ_j]
                  × φ((y_i - X_i β - δ ξ_r)/σ_w)/σ_w × φ(ξ_r)
"""
function factor_model(θ::AbstractVector{T}, X::AbstractMatrix, Xfac::AbstractMatrix,
                      Meas::AbstractMatrix, y::AbstractVector, R::Integer) where T<:Real
    # Dimensions
    K = size(X, 2)
    L = size(Xfac, 2)
    J = size(Meas, 2)
    N = length(y)

    # Unpack parameters
    α, β, γ, δ, σ = unpack_params(θ, K, L, J)

    # Quadrature nodes (ξ) and weights (ω)
    ξ, ω = lgwt(R, -5, 5)

    # Parts that do not change across quadrature points
    XFα = Xfac * α        # N×J
    Xβ  = X * β           # N
    sdM = sqrt.(σ[1:J].^2) # |σ_j| (the sign of σ is not identified)
    sdy = sqrt(σ[end]^2)

    like  = zeros(T, N)
    Mlike = zeros(T, N, J)
    for r in 1:R
        # Measurement equations
        for j in 1:J
            Mres = Meas[:, j] .- XFα[:, j] .- γ[j] * ξ[r]
            Mlike[:, j] = pdf.(Normal(0, 1), Mres ./ sdM[j]) ./ sdM[j]
        end
        # Wage equation
        Yres  = y .- Xβ .- δ * ξ[r]
        Ylike = pdf.(Normal(0, 1), Yres ./ sdy) ./ sdy
        # Add this quadrature point (element-wise products!)
        like .+= ω[r] .* vec(prod(Mlike; dims=2)) .* Ylike .* pdf(Normal(0, 1), ξ[r])
    end

    return -sum(log.(like))
end

"""
    get_starting_values(X, y, Xfac, asvabs)

Starting values: OLS for α (each ASVAB on Xfac) and β (wage on X), 0.5 for the
loadings γ and δ (all-zero loadings are a stationary point because the
likelihood is symmetric in their sign), and 0.5 for every σ.
"""
function get_starting_values(X::AbstractMatrix, y::AbstractVector,
                             Xfac::AbstractMatrix, asvabs::AbstractMatrix)
    J  = size(asvabs, 2)
    α0 = Xfac \ asvabs          # L×J
    β0 = X \ y
    γ0 = fill(0.5, J)
    δ0 = 0.5
    σ0 = fill(0.5, J + 1)
    return vcat(vec(α0), β0, γ0, δ0, σ0)
end

"""
    run_estimation(df, start_vals; R=41, show_trace=true)

Estimate the factor model by MLE with Newton's method (ForwardDiff Hessian).
Returns (θ̂, se, loglike), where loglike is the maximized log likelihood.

Note on R: with 6 test scores the posterior of ξ is narrow, so few nodes give a
poor approximation of the integral. In my checks the estimates were clearly off
with R = 9 and stopped changing from about R = 41, so I use R = 41.
"""
function run_estimation(df::DataFrame, start_vals::Vector; R::Integer=41, show_trace::Bool=true)
    X, y, Xfac, asvabs = prepare_factor_matrices(df)
    J = size(asvabs, 2)
    obj = θ -> factor_model(θ, X, Xfac, asvabs, y, R)

    td = TwiceDifferentiable(obj, start_vals; autodiff = AD_BACKEND)
    result = optimize(td, start_vals, Newton(linesearch = Optim.LineSearches.BackTracking()),
                      Optim.Options(g_tol = 1e-5, iterations = 1_000,
                                    show_trace = show_trace, show_every = 1))
    println("Converged: ", Optim.converged(result), "  (iterations: ", Optim.iterations(result), ")")

    θ̂ = copy(Optim.minimizer(result))

    # Normalize the sign of the factor: (γ, δ) and (-γ, -δ) give the same
    # likelihood, so I choose the sign with positive average ASVAB loading.
    K, L = size(X, 2), size(Xfac, 2)
    iγ = J*L+K+1 : J*L+K+J+1               # γ and δ
    if mean(θ̂[J*L+K+1 : J*L+K+J]) < 0
        θ̂[iγ] .= -θ̂[iγ]
    end
    θ̂[end-J:end] .= abs.(θ̂[end-J:end])     # report σ as positive

    # Standard errors from the inverse Hessian of the negative log likelihood
    H  = ForwardDiff.hessian(obj, θ̂)
    se = sqrt.(diag(inv(Symmetric(H))))

    return θ̂, se, -Optim.minimum(result)
end

"""
    format_results(θ, se, loglike, asvabs)

DataFrame with columns equation, variable, coefficient, std_error, z_stat.
Rows follow the order of θ; the last row holds the log likelihood.
"""
function format_results(θ::Vector, se::Vector, loglike::Real, asvabs::Matrix)
    J = size(asvabs, 2)
    asvab_names = J == length(ASVAB_COLS) ? string.(ASVAB_COLS) : ["asvab$j" for j in 1:J]
    xfac_names  = ["black", "hispanic", "female", "constant"]
    x_names     = ["black", "hispanic", "female", "schoolt", "gradHS", "grad4yr", "constant"]

    equation = String[]
    variable = String[]
    for j in 1:J, l in xfac_names              # α (column by column)
        push!(equation, asvab_names[j]); push!(variable, l)
    end
    for k in x_names                            # β
        push!(equation, "logwage"); push!(variable, k)
    end
    for j in 1:J                                # γ
        push!(equation, asvab_names[j]); push!(variable, "factor loading (gamma)")
    end
    push!(equation, "logwage"); push!(variable, "factor loading (delta)")   # δ
    for j in 1:J                                # σ_j
        push!(equation, asvab_names[j]); push!(variable, "sigma")
    end
    push!(equation, "logwage"); push!(variable, "sigma")                    # σ_w

    @assert length(equation) == length(θ) == length(se) "names do not match θ"

    out = DataFrame(equation = equation, variable = variable,
                    coefficient = θ, std_error = se, z_stat = θ ./ se)
    push!(out, ("model", "log likelihood", loglike, NaN, NaN))
    return out
end
