#===========================================================
 ECON 6343: Econometrics III — Problem Set 3
 Author: Prodip Chakraborty 
===========================================================#

using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables
using Test

cd(@__DIR__)
include("PS3_Prodip_source.jl")

#---------------------------------------------------
# Helpers
#---------------------------------------------------

# MNL probabilities written exactly as in the problem set, one person at a time
function naive_mlogit_probs(θ, X, Z, J)
    K = size(X, 2); N = size(X, 1)
    β = [reshape(θ[1:end-1], K, J-1) zeros(K)]
    γ = θ[end]
    P = zeros(N, J)
    for i = 1:N
        e = [exp(dot(X[i, :], β[:, j]) + γ * (Z[i, j] - Z[i, J])) for j = 1:J]
        P[i, :] = e ./ sum(e)
    end
    return P
end

# Nested-logit probabilities written exactly as in the problem set
function naive_nested_probs(θ, X, Z, nests, J)
    K = size(X, 2); N = size(X, 1)
    βW, βB = θ[1:K], θ[K+1:2K]
    λW, λB, γ = θ[2K+1], θ[2K+2], θ[2K+3]
    WC, BC = nests
    P = zeros(N, J)
    for i = 1:N
        eW = Dict(j => exp((dot(X[i, :], βW) + γ * (Z[i, j] - Z[i, J])) / λW) for j in WC)
        eB = Dict(j => exp((dot(X[i, :], βB) + γ * (Z[i, j] - Z[i, J])) / λB) for j in BC)
        DW, DB = sum(values(eW)), sum(values(eB))
        den = 1 + DW^λW + DB^λB
        for j in WC; P[i, j] = eW[j] * DW^(λW - 1) / den; end
        for j in BC; P[i, j] = eB[j] * DB^(λB - 1) / den; end
        P[i, J] = 1 / den
    end
    return P
end

function fake_data(N; J = 8, seed = 42)
    rng = MersenneTwister(seed)
    X = [rand(rng, 35:45, N) rand(rng, 0:1, N) rand(rng, 0:1, N)]
    Z = 1.8 .+ 0.5 .* randn(rng, N, J)
    y = rand(rng, 1:J, N)
    return X, Z, y
end

function draw_choices(P, rng)
    N, J = size(P)
    y = zeros(Int, N)
    for i = 1:N
        u = rand(rng); c = cumsum(P[i, :])
        y[i] = min(searchsortedfirst(c, u), J)
    end
    return y
end

const NESTS = [[1, 2, 3], [4, 5, 6, 7]]
const J = 8

@testset "PS3 unit tests" begin

    @testset "choice_matrix" begin
        y = [1, 3, 8, 3]
        B = choice_matrix(y, J)
        @test size(B) == (4, 8)
        @test all(sum(B, dims = 2) .== 1)
        @test B[1, 1] == 1 && B[2, 3] == 1 && B[3, 8] == 1 && B[4, 3] == 1
        @test sum(B) == 4
    end

    @testset "mlogit_probs" begin
        X, Z, _ = fake_data(50)
        θ = [0.05 .* randn(MersenneTwister(1), 21); 0.7]
        P = mlogit_probs(θ, X, Z, J)
        @test size(P) == (50, 8)
        @test all(P .>= 0)
        @test all(abs.(sum(P, dims = 2) .- 1) .< 1e-12)
        @test P ≈ naive_mlogit_probs(θ, X, Z, J)             # matches the formula
        # θ = 0 → every alternative equally likely
        @test mlogit_probs(zeros(22), X, Z, J) ≈ fill(1 / 8, 50, 8)
        # γ really enters: changing γ changes the probabilities
        θ2 = copy(θ); θ2[end] = -0.7
        @test !(mlogit_probs(θ2, X, Z, J) ≈ P)
    end

    @testset "mlogit_with_Z" begin
        X, Z, y = fake_data(100)
        @test mlogit_with_Z(zeros(22), X, Z, y) ≈ 100 * log(8)
        θ = [0.03 .* randn(MersenneTwister(3), 21); 0.4]
        Pn = naive_mlogit_probs(θ, X, Z, J)
        @test mlogit_with_Z(θ, X, Z, y) ≈ -sum(log(Pn[i, y[i]]) for i in 1:100)
        θ2 = copy(θ); θ2[end] = 2.0
        @test mlogit_with_Z(θ2, X, Z, y) != mlogit_with_Z(θ, X, Z, y)
    end

    @testset "nested_logit_probs" begin
        X, Z, _ = fake_data(50)
        θ = [0.02, 0.3, -0.5, 0.01, -0.2, 0.4, 0.6, 0.8, 0.9]
        P = nested_logit_probs(θ, X, Z, NESTS, J)
        @test size(P) == (50, 8)
        @test all(P .>= 0)
        @test all(abs.(sum(P, dims = 2) .- 1) .< 1e-12)
        @test P ≈ naive_nested_probs(θ, X, Z, NESTS, J)      # matches the formula
        θ1 = copy(θ); θ1[7] = 1.0; θ1[8] = 1.0
        βW, βB = θ1[1:3], θ1[4:6]
        θmnl = [βW; βW; βW; βB; βB; βB; βB; θ1[9]]
        @test nested_logit_probs(θ1, X, Z, NESTS, J) ≈ mlogit_probs(θmnl, X, Z, J)
    end

    @testset "nested_logit_with_Z" begin
        X, Z, y = fake_data(100)
        θ = [0.02, 0.3, -0.5, 0.01, -0.2, 0.4, 0.6, 0.8, 0.9]
        Pn = naive_nested_probs(θ, X, Z, NESTS, J)
        @test nested_logit_with_Z(θ, X, Z, y, NESTS) ≈ -sum(log(Pn[i, y[i]]) for i in 1:100)
    end

    @testset "optimize_mlogit (simulated data)" begin
        rng = MersenneTwister(2026)
        N = 20_000
        X = [rand(rng, 35:45, N) rand(rng, 0:1, N) rand(rng, 0:1, N)]
        Z = 1.8 .+ 0.5 .* randn(rng, N, J)
        θtrue = [vcat([[0.01 * randn(rng), 0.5 * randn(rng), 0.5 * randn(rng)] for _ = 1:7]...); 1.0]
        y = draw_choices(mlogit_probs(θtrue, X, Z, J), rng)

        Random.seed!(1234)
        θ̂ = optimize_mlogit(X, Z, y)
        @test length(θ̂) == 22
        @test mlogit_with_Z(θ̂, X, Z, y) <= mlogit_with_Z(θtrue, X, Z, y)
        @test abs(θ̂[end] - 1.0) < 0.25
    end

    @testset "optimize_nested_logit (simulated data)" begin
        rng = MersenneTwister(7)
        N = 10_000
        X = [rand(rng, 35:45, N) rand(rng, 0:1, N) rand(rng, 0:1, N)]
        Z = 1.8 .+ 0.5 .* randn(rng, N, J)
        θtrue = [0.02, 0.4, -0.3, 0.01, -0.2, 0.3, 0.5, 0.7, 1.0]
        y = draw_choices(nested_logit_probs(θtrue, X, Z, NESTS, J), rng)

        Random.seed!(1234)
        θ̂ = optimize_nested_logit(X, Z, y, NESTS)
        @test length(θ̂) == 9
        # MLE property: fitted likelihood at least as high as at the truth
        @test nested_logit_with_Z(θ̂, X, Z, y, NESTS) <= nested_logit_with_Z(θtrue, X, Z, y, NESTS)
    end

    @testset "load_data" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS3-gev/nlsw88w.csv"
        df, X, Z, y = load_data(url)
        @test size(X) == (2237, 3)
        @test size(Z) == (2237, 8)
        @test length(y) == 2237
        @test sort(unique(y)) == collect(1:8)
    end

    @testset "allwrap (full run on the real data)" begin
        # runs both estimations end to end without error
        @test allwrap() === nothing
    end
end
