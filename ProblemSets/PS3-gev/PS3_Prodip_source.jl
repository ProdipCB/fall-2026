#===========================================================
 ECON 6343: Econometrics III — Problem Set 3
 Author: Prodip Chakraborty
===========================================================#

#---------------------------------------------------
# Data loading
#---------------------------------------------------
function load_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occupation
    return df, X, Z, y
end

function choice_matrix(y, J)
    N = length(y)
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end
    return bigY
end

#---------------------------------------------------
# Question 1
#---------------------------------------------------

function mlogit_probs(θ, X, Z, J)
    α = θ[1:end-1]
    γ = θ[end]
    K = size(X, 2)
    N = size(X, 1)

    bigα = [reshape(α, K, J-1) zeros(K)]      # K × J, last column = 0

    T = promote_type(eltype(X), eltype(θ))
    num = zeros(T, N, J)
    for j = 1:J
        num[:, j] = exp.(X * bigα[:, j] .+ γ .* (Z[:, j] .- Z[:, J]))
    end
    dem = sum(num, dims = 2)
    return num ./ dem
end

function mlogit_with_Z(θ, X, Z, y)
    J = size(Z, 2)
    bigY = choice_matrix(y, J)
    P = mlogit_probs(θ, X, Z, J)
    return -sum(bigY .* log.(P))
end

#---------------------------------------------------
# Question 3
#---------------------------------------------------

function nested_logit_probs(θ, X, Z, nesting_structure, J)
    K = size(X, 2)
    N = size(X, 1)
    α = θ[1:end-3]              # 2K elements: β_WC, β_BC
    λ = θ[end-2:end-1]          # λ_WC, λ_BC
    γ = θ[end]

   
    bigα = zeros(eltype(θ), K, J)
    for j in nesting_structure[1]
        bigα[:, j] = α[1:K]
    end
    for j in nesting_structure[2]
        bigα[:, j] = α[K+1:2K]
    end

    T = promote_type(eltype(X), eltype(θ))
    lidx = zeros(T, N, J)       # exp(v_ij / λ_n)
    num  = zeros(T, N, J)
    dem  = zeros(T, N)

    for j = 1:J
        if j in nesting_structure[1]
            lidx[:, j] = exp.((X * bigα[:, j] .+ γ .* (Z[:, j] .- Z[:, J])) ./ λ[1])
        elseif j in nesting_structure[2]
            lidx[:, j] = exp.((X * bigα[:, j] .+ γ .* (Z[:, j] .- Z[:, J])) ./ λ[2])
        else
            lidx[:, j] = ones(T, N)                 # Other: index normalized to 0
        end
    end

    D_WC = vec(sum(lidx[:, nesting_structure[1]], dims = 2))
    D_BC = vec(sum(lidx[:, nesting_structure[2]], dims = 2))

    for j = 1:J
        if j in nesting_structure[1]
            num[:, j] = lidx[:, j] .* D_WC .^ (λ[1] - 1)
        elseif j in nesting_structure[2]
            num[:, j] = lidx[:, j] .* D_BC .^ (λ[2] - 1)
        else
            num[:, j] = lidx[:, j]
        end
        dem .+= num[:, j]           # = 1 + D_WC^λ_WC + D_BC^λ_BC
    end

    return num ./ dem
end

# Negative log-likelihood (objective to minimize)
function nested_logit_with_Z(θ, X, Z, y, nesting_structure)
    J = size(Z, 2)
    bigY = choice_matrix(y, J)
    P = nested_logit_probs(θ, X, Z, nesting_structure, J)
    return -sum(bigY .* log.(P))
end

#---------------------------------------------------
# Optimization Functions
#---------------------------------------------------

function optimize_mlogit(X, Z, y)
    # Starting values: 21 β's + 1 γ
    startvals = [2 * rand(7 * size(X, 2)) .- 1; 0.1]
    result = optimize(θ -> mlogit_with_Z(θ, X, Z, y), startvals, LBFGS(),
                      Optim.Options(g_tol = 1e-5, iterations = 100_000, show_trace = false))
    return result.minimizer
end

function optimize_nested_logit(X, Z, y, nesting_structure)
    # Starting values: 6 β's + 2 λ's + 1 γ
    startvals = [2 * rand(2 * size(X, 2)) .- 1; 1.0; 1.0; 0.1]
    result = optimize(θ -> nested_logit_with_Z(θ, X, Z, y, nesting_structure),
                      startvals, LBFGS(),
                      Optim.Options(g_tol = 1e-5, iterations = 100_000, show_trace = false))
    return result.minimizer
end

#---------------------------------------------------
# Question 4
#---------------------------------------------------
function allwrap()
    Random.seed!(1234)

    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS3-gev/nlsw88w.csv"
    df, X, Z, y = load_data(url)

    println("Data loaded successfully!")
    println("Sample size: ", size(X, 1))
    println("Number of covariates in X: ", size(X, 2))
    println("Number of alternatives: ", size(Z, 2))

    # Question 1: multinomial logit
    println("\n=== MULTINOMIAL LOGIT RESULTS ===")
    theta_hat_mle = optimize_mlogit(X, Z, y)
    println("Estimates: ", theta_hat_mle)
    println("γ̂ = ", theta_hat_mle[end])

    # Question 3: nested logit
    println("\n=== NESTED LOGIT RESULTS ===")
    nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]  # WC and BC occupations
    nlogit_theta_hat = optimize_nested_logit(X, Z, y, nesting_structure)
    println("Estimates [β_WC; β_BC; λ_WC; λ_BC; γ]: ", nlogit_theta_hat)

    return nothing
end
