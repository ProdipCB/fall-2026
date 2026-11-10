#==================================================================================
Problem Set 8 - Unit Tests
ECON 6343: Econometrics III
Author: Prodip Chakraborty
==================================================================================#

using Test
cd(@__DIR__)
include("PS8_Chakraborty_source.jl")

const URL = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS8-factor/nlsy.csv"
const SRC = isfile("nlsy.csv") ? "nlsy.csv" : URL

# Simulate a data set from the factor model with known parameters (used below)
function simulate_factor_data(N::Integer; seed::Integer = 1234)
    rng = MersenneTwister(seed)
    J = 6
    black    = Float64.(rand(rng, N) .< 0.25)
    hispanic = Float64.((rand(rng, N) .< 0.25) .& (black .== 0))
    female   = Float64.(rand(rng, N) .< 0.5)
    schoolt  = 10 .* rand(rng, N)
    gradHS   = Float64.(rand(rng, N) .< 0.9)
    grad4yr  = Float64.((rand(rng, N) .< 0.4) .& (gradHS .== 1))
    ξ = randn(rng, N)

    Xfac = hcat(black, hispanic, female, ones(N))
    X    = hcat(black, hispanic, female, schoolt, gradHS, grad4yr, ones(N))
    α = [-0.8 -0.6 -0.8 -0.4 -0.7 -0.9;
         -0.5 -0.4 -0.6 -0.4 -0.5 -0.7;
          0.0  0.2  0.2  0.2  0.3  0.0;
          0.3  0.2  0.2  0.0  0.1  0.3]                 # L×J
    β = [-0.2, -0.1, -0.15, 0.01, 0.1, 0.25, 2.0]
    γ = [0.8, 0.5, 0.8, 0.6, 0.7, 0.6]
    δ = 0.1
    σ = [0.5, 0.8, 0.45, 0.7, 0.55, 0.6, 0.45]

    asv = Xfac * α .+ ξ * γ' .+ randn(rng, N, J) .* σ[1:J]'
    logwage = X * β .+ δ .* ξ .+ σ[end] .* randn(rng, N)

    df = DataFrame(black = black, hispanic = hispanic, female = female, schoolt = schoolt,
                   gradHS = gradHS, grad4yr = grad4yr, logwage = logwage)
    for (j, c) in enumerate(ASVAB_COLS)
        df[!, c] = asv[:, j]
    end
    θtrue = vcat(vec(α), β, γ, δ, σ)
    return df, θtrue
end

df = load_data(SRC)
N  = nrow(df)

@testset "PS8 unit tests" begin

    @testset "lgwt quadrature" begin
        x, w = lgwt(41, -5, 5)
        @test length(x) == 41 && length(w) == 41
        @test all(-5 .<= x .<= 5)
        @test sum(w) ≈ 10 atol = 1e-10                         # ∫ 1 dx on [-5,5]
        @test sum(w .* x .^ 2) ≈ 250 / 3 rtol = 1e-10           # ∫ x² dx on [-5,5]
        @test sum(w .* pdf.(Normal(0, 1), x)) ≈ 1 atol = 1e-5   # ∫ φ ≈ 1
        @test sum(w .* x .^ 2 .* pdf.(Normal(0, 1), x)) ≈ 1 atol = 1e-4  # Var(ξ) = 1
    end

    @testset "Q1: load_data and base regression" begin
        @test df isa DataFrame
        @test N == 2438
        for c in [:logwage, :black, :hispanic, :female, :schoolt, :gradHS, :grad4yr]
            @test c in propertynames(df)
        end
        @test all(c in propertynames(df) for c in ASVAB_COLS)
        est = estimate_base_regression(df)
        @test length(coef(est)) == 7
        # OLS coefficients equal the normal-equation solution
        Xb = hcat(ones(N), Matrix{Float64}(df[:, [:black, :hispanic, :female, :schoolt, :gradHS, :grad4yr]]))
        @test coef(est) ≈ Xb \ Float64.(df.logwage)
        @test coef(est)[end] > 0                    # grad4yr premium is positive
    end

    @testset "Q2: ASVAB correlations" begin
        C = Matrix(compute_asvab_correlations(df))
        @test size(C) == (6, 6)
        @test all(diag(C) .≈ 1)
        @test C ≈ C'
        @test all(-1 .<= C .<= 1)
        @test all(C .> 0)                           # all tests positively related
        @test C[1, 2] ≈ cor(df.asvabAR, df.asvabCS)
    end

    @testset "Q3: full regression" begin
        est = estimate_full_regression(df)
        @test length(coef(est)) == 13
        @test all(isfinite, stderror(est))
    end

    @testset "Q4: PCA" begin
        s = asvab_index(df, PCA)
        @test length(s) == N
        @test abs(mean(s)) < 1e-8                   # PCA centers the data
        @test cor(s, vec(mean(Matrix{Float64}(df[:, ASVAB_COLS]); dims = 2))) > 0.9
        est = estimate_pca_regression(df)
        @test length(coef(est)) == 8
        @test coef(est)[end] > 0
        @test !(:asvabPCA in propertynames(df))     # input DataFrame not modified
    end

    @testset "Q5: Factor Analysis" begin
        f = asvab_index(df, FactorAnalysis)
        @test length(f) == N
        @test cor(f, asvab_index(df, PCA)) > 0.9    # factor and PC are very similar
        est = estimate_factor_regression(df)
        @test length(coef(est)) == 8
        @test coef(est)[end] > 0
        @test !(:asvabFactor in propertynames(df))
    end

    @testset "Q6: prepare_factor_matrices" begin
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        @test size(X) == (N, 7)
        @test size(Xfac) == (N, 4)
        @test size(asvabs) == (N, 6)
        @test length(y) == N
        @test all(X[:, end] .== 1) && all(Xfac[:, end] .== 1)
        @test X[:, 1] == Float64.(df.black)
        @test asvabs[:, 3] == Float64.(df.asvabMK)
    end

    @testset "Q6: unpack_params and starting values" begin
        K, L, J = 7, 4, 6
        θ = collect(1.0:45.0)
        α, β, γ, δ, σ = unpack_params(θ, K, L, J)
        @test size(α) == (L, J) && α[:, 1] == 1:4 && α[:, 6] == 21:24
        @test β == 25:31
        @test γ == 32:37
        @test δ == 38
        @test σ == 39:45
        @test_throws AssertionError unpack_params(θ[1:44], K, L, J)

        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        sv = get_starting_values(X, y, Xfac, asvabs)
        @test length(sv) == 45
        a0, b0, _, _, _ = unpack_params(sv, K, L, J)
        @test a0 ≈ Xfac \ asvabs
        @test b0 ≈ X \ y
    end

    @testset "Q6: factor_model likelihood" begin
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        K, L, J = 7, 4, 6
        θ = get_starting_values(X, y, Xfac, asvabs)
        nll = factor_model(θ, X, Xfac, asvabs, y, 41)
        @test nll isa Real && isfinite(nll)

        # With γ = δ = 0 the factor drops out, so the likelihood is the product
        # of independent normal densities (up to quadrature error of ∫φ ≈ 1).
        θ0 = copy(θ); θ0[J*L+K+1 : J*L+K+J+1] .= 0
        α, β, _, _, σ = unpack_params(θ0, K, L, J)
        nll_indep = -sum(logpdf.(Normal.(0, σ[1:J]'), asvabs .- Xfac * α)) -
                     sum(logpdf.(Normal(0, σ[end]), y .- X * β))
        @test factor_model(θ0, X, Xfac, asvabs, y, 41) ≈ nll_indep rtol = 1e-6

        # Flipping the sign of all loadings does not change the likelihood
        θneg = copy(θ); θneg[J*L+K+1 : J*L+K+J+1] .*= -1
        @test factor_model(θneg, X, Xfac, asvabs, y, 41) ≈ nll

        # Sign of σ does not matter
        θs = copy(θ); θs[end] *= -1
        @test factor_model(θs, X, Xfac, asvabs, y, 41) ≈ nll

        # Hand check for one person
        i = 1
        α, β, γ, δ, σ = unpack_params(θ, K, L, J)
        ξ, ω = lgwt(41, -5, 5)
        Li = sum(ω[r] * prod(pdf(Normal(0, σ[j]), asvabs[i, j] - Xfac[i, :]' * α[:, j] - γ[j] * ξ[r]) for j in 1:J) *
                 pdf(Normal(0, σ[end]), y[i] - X[i, :]' * β - δ * ξ[r]) * pdf(Normal(0, 1), ξ[r]) for r in 1:41)
        @test factor_model(θ, X[i:i, :], Xfac[i:i, :], asvabs[i:i, :], y[i:i], 41) ≈ -log(Li)

        # Works with ForwardDiff (needed for Newton)
        g = ForwardDiff.gradient(t -> factor_model(t, X, Xfac, asvabs, y, 41), θ)
        @test length(g) == 45 && all(isfinite, g)
    end

    @testset "Q6: format_results" begin
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        θ = get_starting_values(X, y, Xfac, asvabs)
        out = format_results(θ, ones(45), -100.0, asvabs)
        @test out isa DataFrame
        @test nrow(out) == 46
        @test names(out) == ["equation", "variable", "coefficient", "std_error", "z_stat"]
        @test out.equation[1] == "asvabAR" && out.variable[4] == "constant"
        @test out.equation[25] == "logwage" && out.variable[25] == "black"
        @test out.variable[38] == "factor loading (delta)"
        @test out.coefficient[end] == -100.0
    end

    @testset "Q6: MLE recovers true parameters (simulated data)" begin
        dfsim, θtrue = simulate_factor_data(3000)
        X, y, Xfac, asvabs = prepare_factor_matrices(dfsim)
        sv = get_starting_values(X, y, Xfac, asvabs)
        θ̂, se, ll = run_estimation(dfsim, sv; R = 41, show_trace = false)
        @test length(θ̂) == 45 && length(se) == 45
        @test all(se .> 0) && all(isfinite, se)
        @test isfinite(ll)
        @test ll ≈ -factor_model(θ̂, X, Xfac, asvabs, y, 41)
        # Estimates within 4 standard errors of the truth
        @test all(abs.(θ̂ .- θtrue) .< 4 .* se)
        # Optimum: higher likelihood than the starting values
        @test ll > -factor_model(sv, X, Xfac, asvabs, y, 41)
    end
end
