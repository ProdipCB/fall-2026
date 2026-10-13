#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 5
# Prodip Chakraborty
#---------------------------------------------------

using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

include("create_grids.jl")
include("PS5_Chakraborty_source.jl")

# small fake data set with the same fields as load_dynamic_data().
# To keep the tests fast it uses only the first 3 route-usage bins
# (the full grid has 101), so solving the model is about 30 times cheaper.
function make_test_data(; β = 0.9, N = 50, T = 3, nz = 3, seed = 1234)
    Random.seed!(seed)
    zval, zbin, xval, xbin, xtran = create_grids()
    zval   = zval[1:nz]
    zbin   = nz
    xtran  = xtran[1:nz*xbin, :]           # rows for the first nz route-usage bins
    Xstate = rand(1:40, N, T)              # mileage bins
    X      = xval[Xstate]                  # mileage values that match the bins
    Zstate = rand(1:zbin, N)
    B      = rand(0:1, N)
    Y      = rand(0:1, N, T)
    return (Y = Y, X = X, B = B, Xstate = Xstate, Zstate = Zstate,
            N = N, T = T,
            xval = xval, xbin = xbin, zval = zval, zbin = zbin, xtran = xtran,
            β = β)
end

θ_test = [2.0, -0.15, 1.0]

@testset "PS5 unit tests" begin

    #-----------------------------------------------
    # create_grids (provided code, Question 3b)
    #-----------------------------------------------
    @testset "create_grids" begin
        zval, zbin, xval, xbin, xtran = create_grids()
        @test zbin == 101
        @test xbin == 201
        @test zval[1] ≈ 0.25 && zval[end] ≈ 1.25
        @test xval[1] ≈ 0.0 && xval[end] ≈ 25.0
        @test size(xtran) == (zbin * xbin, xbin)
        @test all(xtran .>= -1e-12)                                     # no negative probabilities
        @test all(isapprox.(sum(xtran, dims = 2), 1.0; atol = 1e-8))   # Markov rows
    end

    #-----------------------------------------------
    # Question 2: static logit
    #-----------------------------------------------
    @testset "estimate_static_model" begin
        Random.seed!(42)
        n   = 20_000
        odo = 10 .* rand(n)
        br  = Float64.(rand(0:1, n))
        p   = 1 ./ (1 .+ exp.(-(θ_test[1] .+ θ_test[2] .* odo .+ θ_test[3] .* br)))
        y   = Float64.(rand(n) .< p)
        df  = DataFrame(Y = y, Odometer = odo, Branded = br)

        fit = estimate_static_model(df)
        @test length(coef(fit)) == 3
        @test isapprox(coef(fit), θ_test; atol = 0.15)   # recovers true θ
    end

    #-----------------------------------------------
    # Question 3c: future values
    #-----------------------------------------------
    @testset "compute_future_value!" begin
        d  = make_test_data()
        FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
        compute_future_value!(FV, θ_test, d)

        @test size(FV) == (d.zbin * d.xbin, 2, d.T + 1)
        @test all(FV[:, :, d.T + 1] .== 0)          # terminal condition
        @test all(isfinite.(FV))

        # last period: FV = β·log(1 + exp(flow utility)), since FV at T+1 is 0
        for (x, z, b) in [(1, 1, 0), (50, 2, 1), (201, 3, 1)]
            row = x + (z-1)*d.xbin
            v1  = θ_test[1] + θ_test[2]*d.xval[x] + θ_test[3]*b
            @test FV[row, b+1, d.T] ≈ d.β * log(exp(0.0) + exp(v1))
        end

        # one period earlier: check one cell with an explicit sum
        x, z, b, t = 10, 2, 1, d.T - 1
        row  = x + (z-1)*d.xbin
        row0 = 1 + (z-1)*d.xbin
        ev1 = 0.0; ev0 = 0.0
        for j in 1:d.xbin
            ev1 += d.xtran[row, j]  * FV[(z-1)*d.xbin + j, b+1, t+1]
            ev0 += d.xtran[row0, j] * FV[(z-1)*d.xbin + j, b+1, t+1]
        end
        v1 = θ_test[1] + θ_test[2]*d.xval[x] + θ_test[3]*b + ev1
        @test FV[row, b+1, t] ≈ d.β * log(exp(ev0) + exp(v1))

        # with β = 0 all future values are 0
        d0  = make_test_data(β = 0.0)
        FV0 = zeros(d0.zbin * d0.xbin, 2, d0.T + 1)
        compute_future_value!(FV0, θ_test, d0)
        @test all(FV0 .== 0)
    end

    #-----------------------------------------------
    # Question 3d-3f: log likelihood
    #-----------------------------------------------
    @testset "log_likelihood_dynamic" begin
        d   = make_test_data()
        nll = log_likelihood_dynamic(θ_test, d)
        @test isfinite(nll)
        @test nll > 0                               # negative of a log likelihood

        # with β = 0 it must equal the static logit negative log likelihood
        d0 = make_test_data(β = 0.0)
        ll_static = 0.0
        for i in 1:d0.N, t in 1:d0.T
            v  = θ_test[1] + θ_test[2]*d0.X[i,t] + θ_test[3]*d0.B[i]
            P1 = exp(v) / (1 + exp(v))
            ll_static += d0.Y[i,t]*log(P1) + (1 - d0.Y[i,t])*log(1 - P1)
        end
        @test log_likelihood_dynamic(θ_test, d0) ≈ -ll_static

        # with β = 0.9 it must match an explicit computation
        FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
        compute_future_value!(FV, θ_test, d)
        ll = 0.0
        for i in 1:d.N, t in 1:d.T
            row0 = 1 + (d.Zstate[i]-1)*d.xbin
            row1 = d.Xstate[i,t] + (d.Zstate[i]-1)*d.xbin
            ev = 0.0
            for j in 1:d.xbin
                ev += (d.xtran[row1, j] - d.xtran[row0, j]) * FV[row0 + j - 1, d.B[i]+1, t+1]
            end
            v  = θ_test[1] + θ_test[2]*d.X[i,t] + θ_test[3]*d.B[i] + ev
            P1 = exp(v) / (1 + exp(v))
            ll += d.Y[i,t]*log(P1) + (1 - d.Y[i,t])*log(1 - P1)
        end
        @test nll ≈ -ll
    end

    #-----------------------------------------------
    # Question 3e, 3h: estimation with Optim
    #-----------------------------------------------
    @testset "estimate_dynamic_model" begin
        d       = make_test_data()
        θ_start = [0.0, 0.0, 0.0]
        result  = estimate_dynamic_model(d, θ_start)
        @test length(result.minimizer) == 3
        @test result.minimum <= log_likelihood_dynamic(θ_start, d)   # improves on start
    end

    #-----------------------------------------------
    # Questions 1 and 3a: data loading (needs internet)
    #-----------------------------------------------
    @testset "load_static_data" begin
        df_long = load_static_data()
        @test nrow(df_long) == 20_000
        @test all(in(names(df_long)), ["bus_id", "time", "Y", "Odometer", "Branded", "RouteUsage"])
        @test sort(unique(df_long.time)) == collect(1:20)
        @test all(in((0, 1)), df_long.Y)
    end

    @testset "load_dynamic_data" begin
        d = load_dynamic_data()
        @test size(d.Y) == (1000, 20)
        @test size(d.X) == (1000, 20)
        @test size(d.Xstate) == (1000, 20)
        @test length(d.Zstate) == 1000
        @test length(d.B) == 1000
        @test d.β == 0.9
        @test all(1 .<= d.Xstate .<= d.xbin)
        @test all(1 .<= d.Zstate .<= d.zbin)
    end
end
