#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 4 (Mixed Logit Models)
# Prodip Chakraborty
#---------------------------------------------------
import Pkg
haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add("ForwardDiff")

using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM,
      FreqTables, Distributions, ForwardDiff

cd(@__DIR__)
include("lgwt.jl")
include("PS4_Chakraborty_source.jl")


#---------------------------------------------------
# Helper: simulate data from the multinomial logit model
#   (same structure as the real data: K = 3, J = 8)
#---------------------------------------------------
function simulate_mlogit(N, θ; seed = 42)
    rng = MersenneTwister(seed)
    K, J = 3, 8
    X = Float64.([rand(rng, 20:45, N) rand(rng, 0:1, N) rand(rng, 0:1, N)])
    Z = 1.5 .+ 0.3 .* randn(rng, N, J)

    bigα = [reshape(θ[1:end-1], K, J-1) zeros(K)]
    γ = θ[end]
    U = X * bigα .+ γ .* (Z .- Z[:, J])
    U .+= -log.(-log.(rand(rng, N, J)))      # add Type I extreme value errors
    y = [argmax(U[i, :]) for i in 1:N]
    return X, Z, y
end

# "True" parameters for the simulation (close to the Question 1 estimates)
θ_true = [ 0.0404,  0.2440, -1.5713,
           0.0433,  0.1469, -2.9591,
           0.1021,  0.7473, -4.1201,
           0.0376,  0.6885, -3.6558,
           0.0205, -0.3584, -4.3769,
           0.1075, -0.5264, -6.1992,
           0.1169, -0.2871, -5.3223,
           1.3075]

X_s, Z_s, y_s = simulate_mlogit(5_000, θ_true)
N_s, K_s = size(X_s)
J_s = size(Z_s, 2)


@testset "PS4 unit tests" begin

    #-----------------------------------------------
    @testset "load_data" begin
        df, X, Z, y = load_data()
        @test size(X) == (28_365, 3)
        @test size(Z) == (28_365, 8)
        @test length(y) == 28_365
        @test sort(unique(y)) == collect(1:8)
        @test !any(ismissing, X) && !any(ismissing, Z)
    end

    #-----------------------------------------------
    @testset "choice_matrix" begin
        y = [1, 3, 2, 3]
        Y = choice_matrix(y, 3)
        @test size(Y) == (4, 3)
        @test all(sum(Y, dims = 2) .== 1)          # one choice per row
        @test Y[2, 3] == 1 && Y[2, 1] == 0
        @test vec(sum(Y, dims = 1)) == [1, 1, 2]   # counts by choice
    end

    #-----------------------------------------------
    @testset "logit_probs" begin
        bigα = [reshape(θ_true[1:end-1], K_s, J_s-1) zeros(K_s)]
        P = logit_probs(bigα, θ_true[end], X_s, Z_s)
        @test size(P) == (N_s, J_s)
        @test all(P .> 0)
        @test all(isapprox.(sum(P, dims = 2), 1.0; atol = 1e-12))

        # all parameters 0 -> every choice has probability 1/J
        P0 = logit_probs(zeros(K_s, J_s), 0.0, X_s, Z_s)
        @test all(isapprox.(P0, 1 / J_s; atol = 1e-12))

        # hand calculation for the first person
        u = [dot(X_s[1, :], bigα[:, j]) + θ_true[end] * (Z_s[1, j] - Z_s[1, J_s]) for j in 1:J_s]
        @test P[1, :] ≈ exp.(u) ./ sum(exp.(u))
    end

    #-----------------------------------------------
    @testset "mlogit_with_Z" begin
        # at θ = 0 the negative log-likelihood is N * log(J)
        @test mlogit_with_Z(zeros(22), X_s, Z_s, y_s) ≈ N_s * log(J_s)

        # positive and finite at the true parameters
        ll = mlogit_with_Z(θ_true, X_s, Z_s, y_s)
        @test isfinite(ll) && ll > 0

        # the true parameters fit better than θ = 0
        @test ll < mlogit_with_Z(zeros(22), X_s, Z_s, y_s)

        # automatic differentiation works and matches finite differences
        f = θ -> mlogit_with_Z(θ, X_s, Z_s, y_s)
        g_ad = ForwardDiff.gradient(f, θ_true)
        h = 1e-5
        g_fd = [(f(θ_true .+ h .* (1:22 .== k)) - f(θ_true .- h .* (1:22 .== k))) / (2h) for k in 1:22]
        @test isapprox(g_ad, g_fd; rtol = 1e-4)
    end

    #-----------------------------------------------
    @testset "optimize_mlogit (simulated data)" begin
        θ_hat, se = optimize_mlogit(X_s, Z_s, y_s)
        @test length(θ_hat) == 22
        @test length(se) == 22
        @test all(isfinite, se) && all(se .> 0)

        # estimates are close to the true values (within 4 standard errors)
        @test all(abs.(θ_hat .- θ_true) .< 4 .* se)
        @test abs(θ_hat[end] - θ_true[end]) < 4 * se[end]

        # gradient is (almost) zero at the estimates
        g = ForwardDiff.gradient(θ -> mlogit_with_Z(θ, X_s, Z_s, y_s), θ_hat)
        @test maximum(abs.(g)) < 1e-2
    end

    #-----------------------------------------------
    @testset "practice_quadrature (Q3a)" begin
        int_density, int_mean = practice_quadrature()
        @test isapprox(int_density, 1.0; atol = 0.01)
        @test isapprox(int_density, 1.0045; atol = 1e-4)
        @test abs(int_mean) < 1e-10
    end

    #-----------------------------------------------
    @testset "lgwt" begin
        nodes, weights = lgwt(7, -4, 4)
        @test length(nodes) == 7 && length(weights) == 7
        @test sum(weights) ≈ 8.0                         # length of [-4, 4]
        @test all(-4 .< nodes .< 4)
        @test sum(weights .* nodes.^2) ≈ 2 * 4^3 / 3      # exact for polynomials
    end

    #-----------------------------------------------
    @testset "variance_quadrature (Q3b)" begin
        var7, var10 = variance_quadrature()
        @test isapprox(var7,  3.2655; atol = 1e-3)
        @test isapprox(var10, 4.0390; atol = 1e-3)
        @test abs(var10 - 4) < abs(var7 - 4)    # more points -> more accurate
    end

    #-----------------------------------------------
    @testset "mc_integrate and practice_monte_carlo (Q3c)" begin
        Random.seed!(1234)
        # constant function: exact for any D
        @test mc_integrate(x -> 1.0, 0, 2, 100) ≈ 2.0
        # ∫_0^1 x dx = 0.5
        @test isapprox(mc_integrate(x -> x, 0, 1, 1_000_000), 0.5; atol = 0.005)

        Random.seed!(1234)
        res = practice_monte_carlo()
        @test haskey(res, 1_000) && haskey(res, 1_000_000)
        v, m, d = res[1_000_000]
        @test isapprox(v, 4.0; atol = 0.05)
        @test isapprox(m, 0.0; atol = 0.02)
        @test isapprox(d, 1.0; atol = 0.01)
    end

    #-----------------------------------------------
    @testset "mixed_logit_quad (Q4)" begin
        nodes, weights = lgwt(7, -4, 4)
        θ_mix = [θ_true; 0.0]                    # σ_γ = exp(0) = 1

        ll = mixed_logit_quad(θ_mix, X_s, Z_s, y_s, nodes, weights)
        @test isfinite(ll) && ll > 0

        # σ_γ -> 0: mixed logit = multinomial logit
        θ_tiny = [θ_true; -30.0]
        @test isapprox(mixed_logit_quad(θ_tiny, X_s, Z_s, y_s, nodes, weights),
                       mlogit_with_Z(θ_true, X_s, Z_s, y_s); rtol = 1e-8)

        # if Z is the same for all choices, γ does not matter,
        # so the mixed logit equals the multinomial logit for any σ_γ
        Z_flat = repeat(Z_s[:, 1], 1, J_s)
        @test isapprox(mixed_logit_quad([θ_true; 0.5], X_s, Z_flat, y_s, nodes, weights),
                       mlogit_with_Z(θ_true, X_s, Z_flat, y_s); rtol = 1e-10)

        # σ_γ changes the likelihood when Z varies
        @test !isapprox(ll, mixed_logit_quad([θ_true; -2.0], X_s, Z_s, y_s, nodes, weights))

        # automatic differentiation works
        g = ForwardDiff.gradient(θ -> mixed_logit_quad(θ, X_s, Z_s, y_s, nodes, weights), θ_mix)
        @test length(g) == 23 && all(isfinite, g)
    end

end
