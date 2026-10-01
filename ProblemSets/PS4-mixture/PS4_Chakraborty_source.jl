#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 4 (Mixed Logit Models)
# Prodip Chakraborty
#---------------------------------------------------


#---------------------------------------------------
# Load the data
#---------------------------------------------------
function load_data()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS4-mixture/nlsw88t.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occ_code
    return df, X, Z, y
end


#---------------------------------------------------
# Helper: N x J matrix of choice dummies (d_itj)
#---------------------------------------------------
function choice_matrix(y, J)
    N = length(y)
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end
    return bigY
end


#---------------------------------------------------
# Helper: logit choice probabilities for one value of γ
#   P_itj = exp(X_it β_j + γ(Z_itj - Z_itJ)) / Σ_k exp(X_it β_k + γ(Z_itk - Z_itJ))
#   with β_J = 0 (normalization)
#---------------------------------------------------
function logit_probs(bigα, γ, X, Z)
    N, J = size(Z)
    T = promote_type(eltype(bigα), typeof(γ), eltype(X))
    num = zeros(T, N, J)
    for j = 1:J
        num[:, j] = exp.(X * bigα[:, j] .+ γ .* (Z[:, j] .- Z[:, J]))
    end
    return num ./ sum(num, dims = 2)
end


#---------------------------------------------------
# Question 1: Multinomial logit with alternative-specific Z
#   θ = [α (K*(J-1) = 21 elements); γ]
#---------------------------------------------------
function mlogit_with_Z(θ, X, Z, y)
    K = size(X, 2)
    J = size(Z, 2)

    α = θ[1:end-1]
    γ = θ[end]
    bigα = [reshape(α, K, J-1) zeros(K)]   # last column = 0 (base choice J)

    bigY = choice_matrix(y, J)
    P = logit_probs(bigα, γ, X, Z)

    return -sum(bigY .* log.(P))           # negative log-likelihood
end

function optimize_mlogit(X, Z, y)
    # Starting values: estimates from PS3 Question 1 (random values are very slow)
    startvals = [ 0.0557,  0.0834, -2.3449,
                  0.0450,  0.7366, -3.1532,
                  0.0926, -0.0842, -4.2733,
                  0.0239,  0.7231, -3.7494,
                  0.0361, -0.6438, -4.2797,
                  0.0853, -1.1714, -6.6787,
                  0.0866, -0.7979, -4.9691,
                 -0.0942]

    # Automatic differentiation (ForwardDiff) for gradient and Hessian
    td = TwiceDifferentiable(θ -> mlogit_with_Z(θ, X, Z, y), startvals;
                             autodiff = Optim.ADTypes.AutoForwardDiff())

    result = optimize(td, startvals, LBFGS(),
                      Optim.Options(g_tol = 1e-5, iterations = 100_000, show_trace = true))

    # Standard errors from the inverse Hessian at the estimates
    H = Optim.hessian!(td, result.minimizer)
    se = sqrt.(diag(inv(H)))
    return result.minimizer, se
end


#---------------------------------------------------
# Question 3a: Quadrature practice with N(0,1)
#---------------------------------------------------
function practice_quadrature()
    println("\n=== Question 3(a): Quadrature with N(0,1) ===")
    d = Normal(0, 1)
    nodes, weights = lgwt(7, -4, 4)

    int_density = sum(weights .* pdf.(d, nodes))
    int_mean    = sum(weights .* nodes .* pdf.(d, nodes))

    println("∫φ(x)dx  = ", int_density, "  (should be ≈ 1)")
    println("∫xφ(x)dx = ", int_mean,    "  (should be ≈ 0)")
    return int_density, int_mean
end


#---------------------------------------------------
# Question 3b: Variance of N(0,2) by quadrature
#---------------------------------------------------
function variance_quadrature()
    println("\n=== Question 3(b): Variance of N(0,2) by quadrature ===")
    σ = 2
    d = Normal(0, σ)

    nodes7, weights7 = lgwt(7, -5σ, 5σ)
    var7 = sum(weights7 .* nodes7.^2 .* pdf.(d, nodes7))

    nodes10, weights10 = lgwt(10, -5σ, 5σ)
    var10 = sum(weights10 .* nodes10.^2 .* pdf.(d, nodes10))

    println("7 points:  ", var7)
    println("10 points: ", var10)
    println("True variance: ", σ^2)
    return var7, var10
end


#---------------------------------------------------
# Question 3c: Monte Carlo integration
#   ∫_a^b f(x)dx ≈ (b-a) * (1/D) * Σ f(X_i),  X_i ~ U[a,b]
#---------------------------------------------------
function mc_integrate(f, a, b, D)
    draws = a .+ (b - a) .* rand(D)
    return (b - a) * mean(f.(draws))
end

function practice_monte_carlo()
    println("\n=== Question 3(c): Monte Carlo integration with N(0,2) ===")
    σ = 2
    d = Normal(0, σ)
    a, b = -5σ, 5σ

    results = Dict{Int, NTuple{3, Float64}}()
    for D in [1_000, 1_000_000]
        int_var  = mc_integrate(x -> x^2 * pdf(d, x), a, b, D)
        int_mean = mc_integrate(x -> x   * pdf(d, x), a, b, D)
        int_dens = mc_integrate(x ->       pdf(d, x), a, b, D)
        println("D = $D:  ∫x²f = $int_var (true 4),  ∫xf = $int_mean (true 0),  ∫f = $int_dens (true 1)")
        results[D] = (int_var, int_mean, int_dens)
    end
    return results
end


#---------------------------------------------------
# Question 4: Mixed logit with quadrature (DO NOT RUN — very slow)
#   γ ~ N(μ_γ, σ_γ²)
#   θ = [α (21 elements); μ_γ; log σ_γ]
#
#   I use log σ_γ so that σ_γ = exp(θ[end]) is always positive.
#   Change of variables: γ_r = μ_γ + σ_γ ξ_r, where ξ_r are the nodes of
#   lgwt(7, -4, 4). Then the weight of node r is ω_r φ(ξ_r), and
#   P_itj = Σ_r ω_r φ(ξ_r) P_itj(γ_r), which is the sum in equation (2).
#
#   With 7 points on [-4, 4], Σ_r ω_r φ(ξ_r) = 1.0045, not exactly 1. I divide
#   the weights by this sum so the probabilities add up to 1. This does not
#   change the estimates, but it makes the log-likelihood comparable to Q1.
#---------------------------------------------------
function mixed_logit_quad(θ, X, Z, y, nodes, weights)
    K = size(X, 2)
    J = size(Z, 2)
    N = length(y)

    α  = θ[1:K*(J-1)]
    μγ = θ[end-1]
    σγ = exp(θ[end])
    bigα = [reshape(α, K, J-1) zeros(K)]

    bigY = choice_matrix(y, J)

    # Quadrature weight times N(0,1) density, normalized to sum to 1
    wφ = weights .* pdf.(Normal(0, 1), nodes)
    wφ = wφ ./ sum(wφ)

    T = promote_type(eltype(θ), eltype(X))
    P = zeros(T, N, J)
    for r in eachindex(nodes)
        γr = μγ + σγ * nodes[r]
        P .+= wφ[r] .* logit_probs(bigα, γr, X, Z)
    end

    return -sum(bigY .* log.(P))
end

function optimize_mixed_logit_quad(X, Z, y, θ_mlogit)
    nodes, weights = lgwt(7, -4, 4)

    # Starting values: α and γ from the multinomial logit (Question 1),
    # and σ_γ = 1 (so log σ_γ = 0)
    startvals = [θ_mlogit[1:end-1]; θ_mlogit[end]; 0.0]

    td = TwiceDifferentiable(θ -> mixed_logit_quad(θ, X, Z, y, nodes, weights), startvals;
                             autodiff = Optim.ADTypes.AutoForwardDiff())

    result = optimize(td, startvals, LBFGS(),
                      Optim.Options(g_tol = 1e-5, iterations = 100_000, show_trace = true))

    H = Optim.hessian!(td, result.minimizer)
    se = sqrt.(diag(inv(H)))
    return result.minimizer, se
end


#---------------------------------------------------
# Question 6: Wrap everything in one function
#---------------------------------------------------
function allwrap()
    println("=== Problem Set 4: Multinomial and Mixed Logit ===")
    Random.seed!(1234)   # so the Monte Carlo results are the same each run

    df, X, Z, y = load_data()
    println("Observations: ", size(X, 1))
    println("Covariates in X: ", size(X, 2))
    println("Alternatives: ", size(Z, 2))

    # Question 1
    println("\n=== Question 1: Multinomial logit ===")
    θ_hat, θ_se = optimize_mlogit(X, Z, y)
    println("Estimates:  ", round.(θ_hat, digits = 4))
    println("Std. errors: ", round.(θ_se, digits = 4))
    println("γ̂ = ", round(θ_hat[end], digits = 4), " (se = ", round(θ_se[end], digits = 4), ")")

    # Question 2
    println("\n=== Question 2: see comments in the script file ===")

    # Question 3
    practice_quadrature()
    variance_quadrature()
    practice_monte_carlo()

    # Question 4 (code is ready, but NOT run: it takes hours)
    println("\n=== Question 4: Mixed logit with quadrature (not run) ===")
    # θ_quad, se_quad = optimize_mixed_logit_quad(X, Z, y, θ_hat)
    # println("Estimates:  ", round.(θ_quad, digits = 4))
    # println("Std. errors: ", round.(se_quad, digits = 4))
    # # σ_γ = exp(log σ_γ); delta method: se(σ_γ) = σ_γ * se(log σ_γ)
    # σ_hat = exp(θ_quad[end])
    # println("μ_γ = ", round(θ_quad[end-1], digits = 4), " (se = ", round(se_quad[end-1], digits = 4), ")")
    # println("σ_γ = ", round(σ_hat, digits = 4), " (se = ", round(σ_hat * se_quad[end], digits = 4), ")")

    println("\n=== Done ===")
end
