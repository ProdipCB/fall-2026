################################################################################
# Problem Set 7 - Unit tests
# ECON 6343: Econometrics III
# Prodip Chakraborty
################################################################################

using Test
include("PS7_Chakraborty_source.jl")

function run_tests()
    @testset "PS7 tests" begin

        # Small fake data used in many tests
        Random.seed!(42)
        N, K, J = 500, 3, 4
        Xs = hcat(ones(N), randn(N, K - 1))
        ys = rand(1:J, N)
        αs = randn(K * (J - 1))

        #----------------------------------------------------------------------
        @testset "Data loading" begin
            url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
            df, Xw, yw = load_data(url)
            @test size(Xw, 2) == 4
            @test size(Xw, 1) == length(yw) == nrow(df)
            @test all(Xw[:, 1] .== 1)
            @test all(in.(Xw[:, 3], Ref([0.0, 1.0])))
            @test yw ≈ log.(df.wage)

            df2, Xo, yo = prepare_occupation_data(df)
            @test !any(ismissing, df2.occupation)
            @test sort(unique(yo)) == collect(1:7)
            @test size(Xo) == (length(yo), 4)
        end

        #----------------------------------------------------------------------
        @testset "Q1: ols_gmm" begin
            β = [1.0, 2.0, -0.5]
            y = Xs * β                      # no error term
            @test ols_gmm(β, Xs, y) ≈ 0 atol = 1e-12
            @test ols_gmm(β .+ 0.1, Xs, y) > 0
            # Equals g' I g with an explicit identity matrix
            b = randn(K); yy = randn(N); g = yy .- Xs * b
            @test ols_gmm(b, Xs, yy) ≈ g' * Matrix(I, N, N) * g
            # Minimizer equals closed-form OLS
            yy2 = Xs * β .+ 0.3 .* randn(N)
            res = optimize(bb -> ols_gmm(bb, Xs, yy2), zeros(K), LBFGS())
            @test res.minimizer ≈ ols_closed_form(Xs, yy2) atol = 1e-6
            @test ols_closed_form(Xs, yy2) ≈ Xs \ yy2
            # Explicit N×N identity weighting matrix gives the same value
            @test ols_gmm(b, Xs, yy; W = Matrix(1.0I, N, N)) ≈ ols_gmm(b, Xs, yy)
        end

        #----------------------------------------------------------------------
        @testset "Helpers: choice_dummies, mlogit_probs" begin
            d = choice_dummies(ys, J)
            @test size(d) == (N, J)
            @test all(sum(d, dims = 2) .== 1)
            @test all(d[i, ys[i]] == 1 for i in 1:N)

            P = mlogit_probs(αs, Xs, J)
            @test size(P) == (N, J)
            @test all(P .> 0)
            @test all(isapprox.(sum(P, dims = 2), 1))
            # α = 0 gives equal probabilities
            @test all(mlogit_probs(zeros(K * (J - 1)), Xs, J) .≈ 1 / J)
            # Stable with very large index values
            @test all(isfinite, mlogit_probs(fill(500.0, K * (J - 1)), Xs, J))
        end

        #----------------------------------------------------------------------
        @testset "Q2: mlogit_mle" begin
            # At α = 0 the -loglik is N log(J)
            @test mlogit_mle(zeros(K * (J - 1)), Xs, ys) ≈ N * log(J)
            @test mlogit_mle(αs, Xs, ys) > 0
            # Matches a hand-written log likelihood
            P = mlogit_probs(αs, Xs, J)
            @test mlogit_mle(αs, Xs, ys) ≈ -sum(log(P[i, ys[i]]) for i in 1:N)
        end

        #----------------------------------------------------------------------
        @testset "Q2: GMM objectives" begin
            # Over-identified objective equals the stacked d - P formula
            d = choice_dummies(ys, J); P = mlogit_probs(αs, Xs, J)
            @test mlogit_gmm_overid(αs, Xs, ys) ≈ sum((d .- P) .^ 2)
            @test mlogit_gmm_overid(αs, Xs, ys) ≥ 0
            # The moment vector has N*J elements
            @test length(d[:] .- P[:]) == N * J
            # Just-identified GMM is ~0 at the MLE
            res = optimize(a -> mlogit_mle(a, Xs, ys), zeros(K * (J - 1)), LBFGS())
            @test mlogit_gmm(res.minimizer, Xs, ys) < 1e-6
        end

        #----------------------------------------------------------------------
        @testset "Q2: mlogit_starting_values" begin
            df = DataFrame(occupation = repeat(1:7, 30))
            Xd = hcat(ones(210), randn(210, 3))
            sv = mlogit_starting_values(df, Xd; J = 7)
            @test length(sv) == 4 * 6
            @test all(isfinite, sv)
        end

        @testset "Q2: GMM with explicit NJ×NJ identity" begin
            Wbig = Matrix(1.0I, N * J, N * J)
            @test mlogit_gmm_overid(αs, Xs, ys; W = Wbig) ≈ mlogit_gmm_overid(αs, Xs, ys)
        end

        @testset "Q3: default_beta and gumbel_choices" begin
            β = default_beta()
            @test size(β) == (3, 4)
            @test all(β[:, end] .== 0)
            # With zero shocks, gumbel_choices picks the largest X_i β_j
            Y0 = gumbel_choices(Xs, β, zeros(N, 4))
            @test Y0 == argmax.(eachrow(Xs * β))
            # Huge shock on alternative 2 forces choice 2
            ε = zeros(N, 4); ε[:, 2] .= 1e6
            @test all(gumbel_choices(Xs, β, ε) .== 2)
        end

        @testset "Output: coef_table" begin
            tab = coef_table(collect(1.0:24.0), 4, 7)
            @test length(split(tab, "\n")) == 4
            @test occursin("const", tab) && occursin("collgrad", tab)
        end

        @testset "Q3: sim_logit (uniform method)" begin
            β = default_beta()
            X, Y = sim_logit(1000, 4, β; seed = 1)
            @test size(X) == (1000, 3)
            @test length(Y) == 1000
            @test all(1 .<= Y .<= 4)
            @test all(X[:, 1] .== 1)
            # Same seed gives the same data
            @test sim_logit(1000, 4, β; seed = 1)[2] == Y
            # Input checks
            @test_throws AssertionError sim_logit(10, 2, randn(3, 2))
            @test_throws AssertionError sim_logit(10, 4, randn(1, 4))
            # Shares match average model probabilities
            Xb, Yb = sim_logit(100_000, 4, β; seed = 2)
            v = Xb * β; Pb = exp.(v) ./ sum(exp.(v), dims = 2)
            @test [mean(Yb .== j) for j in 1:4] ≈ vec(mean(Pb, dims = 1)) atol = 0.01
            # MLE recovers β
            res = optimize(a -> mlogit_mle(a, Xb, Yb), zeros(9), LBFGS())
            @test res.minimizer ≈ vec(β[:, 1:3]) atol = 0.1
        end

        #----------------------------------------------------------------------
        @testset "Q3: sim_logit_w_gumbel" begin
            β = default_beta()
            X, Y = sim_logit_w_gumbel(100_000, 4, β; seed = 3)
            @test all(1 .<= Y .<= 4)
            v = X * β; P = exp.(v) ./ sum(exp.(v), dims = 2)
            @test [mean(Y .== j) for j in 1:4] ≈ vec(mean(P, dims = 1)) atol = 0.01
            res = optimize(a -> mlogit_mle(a, X, Y), zeros(9), LBFGS())
            @test res.minimizer ≈ vec(β[:, 1:3]) atol = 0.1
        end

        #----------------------------------------------------------------------
        @testset "Q5: SMM" begin
            shocks = draw_gumbel_shocks(N, J, 50; seed = 7)
            @test length(shocks) == 50
            @test size(shocks[1]) == (N, J)
            # Same seed gives the same shocks (common random numbers)
            @test draw_gumbel_shocks(N, J, 50; seed = 7)[1] == shocks[1]
            # Objective is deterministic given the shocks
            @test mlogit_smm_overid(αs, Xs, ys, shocks) ==
                  mlogit_smm_overid(αs, Xs, ys, shocks)
            @test mlogit_smm_overid(αs, Xs, ys, shocks) ≥ 0
            # With many draws, simulated frequencies ≈ P, so the SMM
            # objective ≈ GMM objective
            big = draw_gumbel_shocks(N, J, 2000; seed = 8)
            @test mlogit_smm_overid(αs, Xs, ys, big) ≈
                  mlogit_gmm_overid(αs, Xs, ys) rtol = 0.02
        end
    end
end

run_tests()
