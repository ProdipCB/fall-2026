#-----------------------------
# ECON 6343: Econometrics III
# Problem Set 2: unit tests
# Author: Prodip Chakraborty
#-----------------------------

using Test
using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables

include("PS2_Prodip_source.jl")


@testset "build_X" begin
    df = DataFrame(age = [20, 30], race = [1, 2], collgrad = [1, 0])
    X = build_X(df)
    @test size(X) == (2, 4)
    @test X == [1.0 20.0 1.0 1.0; 1.0 30.0 0.0 0.0]
end


#:::::::::::::::::::::::::::::
# Question 1
#:::::::::::::::::::::::::::::
@testset "question 1: f, negf, q1" begin
    for x in (-10.0, -7.378243405529116, -1.0, 0.0, 2.5)
        @test negf([x]) ≈ -f([x])
    end

    
    xknown = -7.378243405529116
    @test f([xknown]) ≈ 964.3133837824209 atol=1e-4
    @test negf([xknown]) ≈ -964.3133837824209 atol=1e-4
    Random.seed!(20260915)
    xstar, fstar = q1()
    @test xstar ≈ xknown atol=1e-3
    @test fstar ≈ 964.3133837824209 atol=1e-3
end


#:::::::::::::::::::::::::
# Question 2
#:::::::::::::::::::::::::
@testset "question 2: ols, q2" begin
    Xtoy = [1.0 0.0; 1.0 1.0; 1.0 2.0]
    ytoy = [2.0, 3.0, 6.0]
    betatoy = [1.0, 2.0]
    @test ols(betatoy, Xtoy, ytoy) ≈ 2.0
    Random.seed!(1)
    N = 100
    age = Float64.(rand(20:60, N))
    white = Float64.(rand(0:1, N))
    collgrad = Float64.(rand(0:1, N))
    beta_true = [0.5, -0.01, 0.30, 0.10]
    Xsim = [ones(N) age white collgrad]
    married = Xsim * beta_true
    df_sim = DataFrame(age = age, white = white, collgrad = collgrad, married = married)
    out = q2(Xsim, married, df_sim)
    @test out.optim ≈ beta_true atol=1e-4
    @test out.closed_form ≈ beta_true atol=1e-8
    @test coef(out.lm) ≈ beta_true atol=1e-8
    @test maximum(abs.(out.optim .- out.closed_form)) < 1e-3
end


#:::::::::::::::::::::::::
# Question 3 & 4
#:::::::::::::::::::::::::
@testset "question 3 & 4: logit_like, q3, q4" begin
    Xtoy = [1.0 0.0; 1.0 1.0]
    ytoy = [1.0, 0.0]
    betatoy = [0.0, 0.0]
    @test logit_like(betatoy, Xtoy, ytoy) ≈ 2*log(2) atol=1e-10
    Random.seed!(2)
    N = 300
    age = Float64.(rand(20:60, N))
    white = Float64.(rand(0:1, N))
    collgrad = Float64.(rand(0:1, N))
    Xsim = [ones(N) age white collgrad]
    beta_true = [0.5, -0.02, 0.8, -0.3]
    p_true = 1 ./ (1 .+ exp.(-Xsim*beta_true))
    married = Float64.(rand(N) .< p_true)
    df_sim = DataFrame(age = age, white = white, collgrad = collgrad, married = married)

    b_logit = q3(Xsim, married)
    p_hat = 1 ./ (1 .+ exp.(-Xsim*b_logit))
    foc = Xsim' * (married .- p_hat)
    @test maximum(abs.(foc)) < 5e-3

    blogit_glm = q4(df_sim)
    @test b_logit ≈ coef(blogit_glm) atol=5e-3
end


#:::::::::::::::::::::::::
# Question 5
#:::::::::::::::::::::::::
@testset "question 5: clean_occupation, mlogit_like, q5" begin
    df = DataFrame(occupation = [1, 7, 8, 9, 13, missing], x = collect(1:6))
    dfm = clean_occupation(df)
    @test nrow(dfm) == 5
    @test dfm.occupation == [1, 7, 7, 7, 7]
    @test nrow(df) == 6  
    Xtoy = [1.0 0.0; 1.0 1.0; 1.0 2.0]
    ytoy = [1, 2, 3]
    alphatoy = zeros(4)    
    @test mlogit_like(alphatoy, Xtoy, ytoy) ≈ 3*log(3) atol=1e-10
    Random.seed!(3)
    N = 400
    age = Float64.(rand(20:60, N))
    white = Float64.(rand(0:1, N))
    collgrad = Float64.(rand(0:1, N))
    Xsim = [ones(N) age white collgrad]
    K, J = 4, 3
    alpha_true = [0.2 -0.1; 0.01 -0.02; 0.5 0.3; -0.2 0.4]   # K x (J-1)
    Xa = Xsim * [alpha_true zeros(K)]
    rowmax = maximum(Xa, dims=2)
    P = exp.(Xa .- rowmax) ./ sum(exp.(Xa .- rowmax), dims=2)
    ysim = [findfirst(cumsum(P[i, :]) .>= rand()) for i in 1:N]
    @test length(unique(ysim)) == J   # sanity check: all 3 categories occur

    alpha_hat = q5(Xsim, ysim)
    @test size(alpha_hat) == (K, J-1)

    Xa_hat = Xsim * [alpha_hat zeros(K)]
    rowmax_hat = maximum(Xa_hat, dims=2)
    P_hat = exp.(Xa_hat .- rowmax_hat) ./ sum(exp.(Xa_hat .- rowmax_hat), dims=2)
    d = [ysim[i] == j for i in 1:N, j in 1:J]
    foc = Xsim' * (d .- P_hat)
    @test maximum(abs.(foc[:, 1:J-1])) < 1e-2
end


@testset "question 5: q5_startval_check" begin
    Random.seed!(3)
    N = 400
    age = Float64.(rand(20:60, N))
    white = Float64.(rand(0:1, N))
    collgrad = Float64.(rand(0:1, N))
    Xsim = [ones(N) age white collgrad]
    K, J = 4, 3
    alpha_true = [0.2 -0.1; 0.01 -0.02; 0.5 0.3; -0.2 0.4]
    Xa = Xsim * [alpha_true zeros(K)]
    rowmax = maximum(Xa, dims=2)
    P = exp.(Xa .- rowmax) ./ sum(exp.(Xa .- rowmax), dims=2)
    ysim = [findfirst(cumsum(P[i, :]) .>= rand()) for i in 1:N]

    results = q5_startval_check(Xsim, ysim)
    @test length(results) == 3
    for r in results
        @test size(r.alpha_mat) == (K, J-1)
    end
    maxgap = maximum(maximum(abs.(results[i].alpha_mat .- results[j].alpha_mat))
                      for i in 1:length(results) for j in i+1:length(results))
    @test maxgap < 1e-1
end



@testset "load_nlsw88" begin
    try
        df = load_nlsw88()
        @test nrow(df) > 0
        @test "white" in names(df)
        @test all(skipmissing(df.white .== (df.race .== 1)))
    catch e
        @info "skipping load_nlsw88 test: could not reach network" exception = e
    end
end


println("PS2_Prodip_tests.jl: all tests completed.")
