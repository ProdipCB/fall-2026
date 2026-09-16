#--------------------------------
# ECON 6343: Econometrics III
# Problem Set 2: source code
# Author: Prodip Chakraborty
#--------------------------------

const NLSW88_URL = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"


function load_nlsw88(url::AbstractString = NLSW88_URL)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    df.white = df.race .== 1
    return df
end


function printpretty(x)
    show(stdout, MIME("text/plain"), x)
    println()
    return nothing
end


function build_X(df)
    return [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
end


#::::::::::::::::::::::::::::
# Question 1
#::::::::::::::::::::::::::::

f(x) = -x[1]^4 - 10x[1]^3 - 2x[1]^2 - 3x[1] - 2


negf(x) = x[1]^4 + 10x[1]^3 + 2x[1]^2 + 3x[1] + 2


function q1(; startval = rand(1))
    result = optimize(negf, startval, LBFGS())
    xstar = Optim.minimizer(result)[1]
    fstar = -Optim.minimum(result)

    println("question 1: optimization summary:")
    println(result)
    println("question 1: argmax of f(x) is ", xstar)
    println("question 1: max of f(x) is ", fstar)

    return xstar, fstar
end


#::::::::::::::::::::::::::::
# Question 2
#::::::::::::::::::::::::::::

function ols(beta, X, y)
    ssr = (y .- X*beta)'*(y .- X*beta)
    return ssr
end


function q2(X, y, df; startval = rand(size(X,2)))
    beta_hat_ols = optimize(b -> ols(b, X, y), startval, LBFGS(),
                            Optim.Options(g_tol=1e-6, iterations=100_000,
                                          show_trace=false))
    b_optim = Optim.minimizer(beta_hat_ols)
    b_closed = inv(X'*X)*X'*y
    b_lm = lm(@formula(married ~ age + white + collgrad), df)
    σ² = sum((y .- X*b_closed).^2)/(size(X,1) - size(X,2))
    vcov_b = σ²*inv(X'*X)
    se = sqrt.(diag(vcov_b))
    println("question 2: OLS estimates from Optim:")
    println(b_optim)
    println("question 2: OLS estimates in closed form, inv(X'X)X'y:")
    println(b_closed)
    println("question 2: largest gap between the two: ",
            maximum(abs.(b_optim .- b_closed)))
    println("question 2: OLS estimates from lm():")
    printpretty(coeftable(b_lm))
    println("question 2: [estimate standard_error] by hand:")
    printpretty([b_closed se])

    return (optim = b_optim, closed_form = b_closed, lm = b_lm, se = se)
end


#::::::::::::::::::::::::::
# Question 3
#::::::::::::::::::::::::::

function logit_like(beta, X, y)
    Xb = X*beta
    logdenom = max.(Xb, 0) .+ log1p.(exp.(-abs.(Xb)))
    loglike = sum(y .* Xb .- logdenom)

    return -loglike
end


function q3(X, y; startval = rand(size(X,2)))
    beta_hat_logit = optimize(b -> logit_like(b, X, y), startval, LBFGS(),
                              Optim.Options(g_tol=1e-6, iterations=100_000,
                                            show_trace=false))
    b_logit = Optim.minimizer(beta_hat_logit)

    println("question 3: logit estimates from Optim:")
    println(b_logit)
    println("question 3: log-likelihood at the optimum: ",
            -Optim.minimum(beta_hat_logit))

    return b_logit
end


#::::::::::::::::::::::::
# question 4
#::::::::::::::::::::::::

function q4(df)
    blogit_glm = glm(@formula(married ~ age + white + collgrad), df,
                     Binomial(), LogitLink())

    println("question 4: logit estimates from glm():")
    printpretty(coeftable(blogit_glm))

    return blogit_glm
end


#:::::::::::::::::::::::::
# question 5
#:::::::::::::::::::::::::

function clean_occupation(df)
    println("question 5: occupation before aggregating:")
    println(freqtable(df, :occupation))
    dfm = dropmissing(df, :occupation)
    dfm[dfm.occupation.==8 ,:occupation] .= 7
    dfm[dfm.occupation.==9 ,:occupation] .= 7
    dfm[dfm.occupation.==10,:occupation] .= 7
    dfm[dfm.occupation.==11,:occupation] .= 7
    dfm[dfm.occupation.==12,:occupation] .= 7
    dfm[dfm.occupation.==13,:occupation] .= 7

    println("question 5: occupation after aggregating:")
    println(freqtable(dfm, :occupation))  # problem solved

    return dfm
end


function mlogit_like(alpha, X, y)
    N = size(X, 1)          # number of observations
    K = size(X, 2)          # number of covariates
    J = maximum(y)          # number of alternatives

    
    alpha_mat = [reshape(alpha, K, J-1) zeros(eltype(alpha), K)]
    Xa = X*alpha_mat
    rowmax = maximum(Xa, dims=2)
    logdenom = rowmax .+ log.(sum(exp.(Xa .- rowmax), dims=2))
    d = [y[i] == j for i in 1:N, j in 1:J]
    loglike = sum(d .* (Xa .- logdenom))

    return -loglike
end


function q5(X, y; startval = zeros(size(X,2)*(maximum(y)-1)))
    K = size(X, 2)
    J = maximum(y)

    alpha_hat_mlogit = optimize(a -> mlogit_like(a, X, y), startval, LBFGS(),
                                Optim.Options(g_tol=1e-5, iterations=100_000,
                                              show_trace=false))
    a_mlogit = Optim.minimizer(alpha_hat_mlogit)
    alpha_mat = reshape(a_mlogit, K, J-1)

    println("question 5: multinomial logit estimates from Optim, as the ",
            K, " x ", J-1, " matrix of alpha_j's")
    println("(rows: intercept, age, white, collgrad; columns: occupation 1 to ",
            J-1, "; occupation ", J, " is the base):")
    printpretty(alpha_mat)
    println("question 5: log-likelihood at the optimum: ",
            -Optim.minimum(alpha_hat_mlogit))
    println("question 5: estimates as the ", length(a_mlogit),
            "-element vector Optim returned:")
    println(a_mlogit)

    return alpha_mat
end


function q5_startval_check(X, y; tol = 1e-3)
    K = size(X, 2)
    J = maximum(y)
    nparm = K*(J-1)

    startvals = [
        ("zeros",    zeros(nparm)),
        ("U[0,1]",   rand(nparm)),
        ("U[-1,1]",  2 .* rand(nparm) .- 1),
    ]

    results = NamedTuple[]
    for (name, sv) in startvals
        opt = optimize(a -> mlogit_like(a, X, y), sv, LBFGS(),
                       Optim.Options(g_tol=1e-5, iterations=100_000,
                                     show_trace=false))
        alpha_mat = reshape(Optim.minimizer(opt), K, J-1)
        loglike = -Optim.minimum(opt)
        push!(results, (startval = name, alpha_mat = alpha_mat, loglike = loglike))
        println("question 5 robustness check: startval = ", name,
                ", log-likelihood = ", loglike)
    end

    maxgap = maximum(maximum(abs.(results[i].alpha_mat .- results[j].alpha_mat))
                      for i in 1:length(results) for j in i+1:length(results))
    println("question 5 robustness check: largest pairwise coefficient gap ",
            "across starting values = ", maxgap)
    if maxgap > tol
        println("question 5 robustness check: WARNING - gap exceeds tol = ", tol,
                "; starting values may be converging to different optima")
    else
        println("question 5 robustness check: estimates agree within tol = ", tol,
                " across all three starting values")
    end

    return results
end


#:::::::::::::::::::::::::
# Question 6
#:::::::::::::::::::::::::

function allwrap()
    Random.seed!(1234)

    #-----------------------------------------------------------------------
    # question 1
    #-----------------------------------------------------------------------
    q1()

    #-----------------------------------------------------------------------
    # questions 2 through 4 use the whole sample
    #-----------------------------------------------------------------------
    df = load_nlsw88()
    X = build_X(df)
    y = df.married .== 1

    q2(X, y, df)
    q3(X, y)
    q4(df)

    #-----------------------------------------------------------------------
    # question 5 needs occupation to be non-missing and aggregated, which
    # changes the number of rows, so X and y are rebuilt
    #-----------------------------------------------------------------------
    dfm = clean_occupation(df)
    Xm = build_X(dfm)
    ym = dfm.occupation

    q5(Xm, ym)
    q5_startval_check(Xm, ym)

    return nothing
end
