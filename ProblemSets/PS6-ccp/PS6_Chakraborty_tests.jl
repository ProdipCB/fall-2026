#=
Problem Set 6 - Unit tests
ECON 6343: Econometrics III
Prodip Chakraborty

Reads in the source code first, then tests every function.
Run with:  julia PS6_Chakraborty_tests.jl
=#

using Test
include(joinpath(@__DIR__, "PS6_Chakraborty_source.jl"))

# Load the real data and fit the flexible logit once (used by several tests)
const DF_LONG = load_and_reshape_data(BUS_URL)
const FLEX    = estimate_flexible_logit(DF_LONG)
const GRIDS   = create_grids()

@testset "PS6 unit tests" begin

    @testset "create_grids / xgrid" begin
        zval, zbin, xval, xbin, xtran = GRIDS
        @test zbin == 101
        @test xbin == 201
        @test size(xtran) == (zbin * xbin, xbin)
        @test all(xtran .>= 0)
        @test all(isapprox.(sum(xtran, dims = 2), 1.0; atol = 1e-10))   # each row is a distribution
        # mileage never goes down: lower-triangular part is zero in every z block
        @test xtran[2, 1] == 0                     # mileage never goes down
        z = 26
        x1, _ = xgrid(zval[z], xval)
        @test x1 == xtran[(z - 1) * xbin + 1 : z * xbin, :]   # block z of xtran = xgrid(zval[z])
    end

    @testset "reshape_to_long (toy data)" begin
        wide = DataFrame(Y1 = [1, 0], Y2 = [0, 1],
                         Odo1 = [1.0, 2.0], Odo2 = [0.0, 3.0],
                         Xst1 = [9, 17], Xst2 = [1, 25],
                         RouteUsage = [0.5, 1.0], Branded = [1, 0], Zst = [26, 76])
        L = reshape_to_long(wide)
        @test nrow(L) == 4
        @test L.bus_id == [1, 1, 2, 2]
        @test L.time == [1, 2, 1, 2]
        @test L.Y == [1, 0, 0, 1]
        @test L.Odometer == [1.0, 0.0, 2.0, 3.0]
        @test L.Xstate == [9, 1, 17, 25]
        @test L.Zst == [26, 26, 76, 76]
        Xs, Zs, Bs = get_state_arrays(L)
        @test Xs == [9 1; 17 25]
        @test Zs == [26, 76]
        @test Bs == [1, 0]
    end

    @testset "load_and_reshape_data (real data)" begin
        @test nrow(DF_LONG) == 20_000
        @test issorted(DF_LONG, [:bus_id, :time])
        @test all(in.(DF_LONG.Y, Ref((0, 1))))
        @test all(1 .<= DF_LONG.Xstate .<= 201)
        @test all(1 .<= DF_LONG.Zst .<= 101)
        # Xstate is the grid index of Odometer
        @test DF_LONG.Odometer ≈ GRIDS[3][DF_LONG.Xstate]
        # Zst is the grid index of RouteUsage
        @test DF_LONG.RouteUsage ≈ GRIDS[1][DF_LONG.Zst]
        Xs, Zs, Bs = get_state_arrays(DF_LONG)
        @test size(Xs) == (1000, 20)
        @test Xs[3, 5] == DF_LONG.Xstate[(3 - 1) * 20 + 5]
    end

    @testset "add_squared_terms" begin
        d = add_squared_terms(DataFrame(Odometer = [2.0], RouteUsage = [0.5], time = [3]))
        @test d.Odometer_sq == [4.0]
        @test d.RouteUsage_sq == [0.25]
        @test d.time_sq == [9]
        @test d.Odo10 ≈ [0.2]
        @test d.Odo10_sq ≈ [0.04]
        @test d.time10 ≈ [0.3]
        @test d.time10_sq ≈ [0.09]
    end

    @testset "estimate_flexible_logit" begin
        @test length(coef(FLEX)) == 128
        @test all(isfinite, coef(FLEX))
        @test all(coef(FLEX) .!= 0)          # no column dropped as collinear
        p = predict(FLEX)
        @test all(0 .< p .< 1)
        # fitted mean should match sample mean (logit with intercept)
        @test mean(p) ≈ mean(DF_LONG.Y) atol = 1e-4
    end

    @testset "construct_state_space matches xtran row order" begin
        zval, zbin, xval, xbin, xtran = GRIDS
        S = construct_state_space(xbin, zbin, xval, zval)
        @test nrow(S) == size(xtran, 1)
        @test all(S.Branded .== 0) && all(S.time .== 0)
        for r in (1, 2, xbin, xbin + 1, 5000, nrow(S))
            @test S.Odometer[r]   == xval[mod1(r, xbin)]
            @test S.RouteUsage[r] == zval[cld(r, xbin)]
        end
        # the row used for a data observation has that observation's states
        i = 123
        r = DF_LONG.Xstate[i] + (DF_LONG.Zst[i] - 1) * xbin
        @test S.Odometer[r] ≈ DF_LONG.Odometer[i]
        @test S.RouteUsage[r] ≈ DF_LONG.RouteUsage[i]
    end

    @testset "softplus and linear_predictor" begin
        for x in (-2.0, 0.0, 1.5, 30.0)
            @test softplus(x) ≈ log(1 + exp(x))
        end
        # very negative x: log(1 + exp(x)) rounds to 0, so compare with exp(x)
        @test softplus(-50.0) ≈ exp(-50.0)
        @test softplus(1000.0) ≈ 1000.0
        @test isfinite(softplus(1000.0))
        η = linear_predictor(FLEX, add_squared_terms(DF_LONG))
        @test 1 ./ (1 .+ exp.(-η)) ≈ predict(FLEX)
    end

    @testset "compute_future_values" begin
        zval, zbin, xval, xbin, xtran = GRIDS
        S  = construct_state_space(xbin, zbin, xval, zval)
        FV = compute_future_values(S, FLEX, xtran, xbin, zbin, 20, 0.9)
        @test size(FV) == (size(xtran, 1), 2, 21)
        @test all(isfinite, FV)
        @test all(FV .>= 0)                  # -β log p0 >= 0
        @test all(FV[:, :, 1] .== 0)         # t = 1 not filled
        @test all(FV[:, :, 21] .== 0)        # terminal period
        # equals -β log(1 - predict()) where that formula is numerically safe
        Sb = copy(S); Sb.time .= 7; Sb.Branded .= 1
        p0 = 1 .- predict(FLEX, add_squared_terms(Sb))
        ok = p0 .> 1e-8
        @test FV[ok, 2, 7] ≈ -0.9 .* log.(p0[ok]) rtol = 1e-6
        # β scales FV linearly
        FV2 = compute_future_values(S, FLEX, xtran, xbin, zbin, 20, 0.45)
        @test FV2 ≈ FV ./ 2
    end

    @testset "compute_fvt1 (toy example)" begin
        # 2 mileage bins, 1 route-usage bin, 2 buses, 2 periods
        xtran = [0.5 0.5;
                 0.0 1.0]
        FV = zeros(2, 2, 3)
        FV[:, 1, 2] = [1.0, 3.0]   # brand 0, period 2
        FV[:, 1, 3] = [2.0, 6.0]   # brand 0, period 3
        FV[:, 2, 2] = [4.0, 4.0]   # brand 1: flat -> difference term is 0
        Xs = [2 1;                 # bus 1: high mileage then low mileage
              2 2]                 # bus 2
        Zs = [1, 1]
        Bs = [0, 1]
        dfl = DataFrame(bus_id = [1, 1, 2, 2], time = [1, 2, 1, 2])
        f = compute_fvt1(dfl, FV, xtran, Xs, Zs, Bs, 2)
        # bus 1, t=1: ([0,1]-[.5,.5])'[1,3] = -0.5 + 1.5 = 1.0
        # bus 1, t=2: row1 = row0 -> 0
        # bus 2: flat FV -> 0
        @test f ≈ [1.0, 0.0, 0.0, 0.0]
        @test length(f) == nrow(dfl)
        # ordering check: bus 1 at t=2 with high mileage uses FV[:,1,3]
        Xs2 = [1 2; 1 1]
        f2 = compute_fvt1(dfl, FV, xtran, Xs2, Zs, [0, 0], 2)
        @test f2 ≈ [0.0, -0.5 * 2.0 + 0.5 * 6.0, 0.0, 0.0]
    end

    @testset "estimate_structural_params and own MLE (simulated)" begin
        Random.seed!(1234)
        n   = 50_000
        odo = 10 .* rand(n)
        br  = rand(0:1, n)
        off = randn(n)
        θ   = [1.5, -0.2, 0.8]
        p   = 1 ./ (1 .+ exp.(-(θ[1] .+ θ[2] .* odo .+ θ[3] .* br .+ off)))
        y   = Int.(rand(n) .< p)
        sim = DataFrame(Y = y, Odometer = odo, Branded = br)
        m   = estimate_structural_params(sim, off)
        @test isapprox(coef(m), θ; atol = 0.08)
        θown, seown = logit_offset_mle(y, hcat(ones(n), odo, br), off)
        @test θown ≈ coef(m) atol = 1e-4
        @test seown ≈ stderror(m) rtol = 1e-3
        # the offset matters: coefficients change without it
        m0 = glm(@formula(Y ~ Odometer + Branded), sim, Binomial(), LogitLink())
        @test !isapprox(coef(m0), coef(m); atol = 1e-3)
    end

    @testset "end-to-end CCP estimates" begin
        zval, zbin, xval, xbin, xtran = GRIDS
        Xs, Zs, Bs = get_state_arrays(DF_LONG)
        S    = construct_state_space(xbin, zbin, xval, zval)
        FV   = compute_future_values(S, FLEX, xtran, xbin, zbin, 20, 0.9)
        fvt1 = compute_fvt1(DF_LONG, FV, xtran, Xs, Zs, Bs, xbin)
        @test length(fvt1) == nrow(DF_LONG)
        @test all(isfinite, fvt1)
        m = estimate_structural_params(DF_LONG, fvt1)
        b = coef(m)
        @test b[1] ≈ 1.94 atol = 0.1     # θ0
        @test b[2] ≈ -0.15 atol = 0.02   # θ1 (odometer)
        @test b[3] ≈ 1.09 atol = 0.1     # θ2 (branded)
    end
end
