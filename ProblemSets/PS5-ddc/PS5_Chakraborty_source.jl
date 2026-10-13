#---------------------------------------------------
# ECON 6343: Econometrics III — Problem Set 5
# Prodip Chakraborty
#---------------------------------------------------

################################################################################
# Question 1: load busdataBeta0.csv and reshape to long format
################################################################################

"""
    load_static_data()

Read busdataBeta0.csv and reshape it from wide (Y1..Y20, Odo1..Odo20) to long
(one row per bus-period). Returns `df_long`.
"""
function load_static_data()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df  = CSV.read(HTTP.get(url).body, DataFrame)
    N   = size(df, 1)

    # bus id
    df = @transform(df, :bus_id = 1:N)

    # reshape the decision variable (Y1-Y20)
    dfy = @select(df, :bus_id, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10,
                      :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20,
                      :RouteUsage, :Branded)
    dfy_long = DataFrames.stack(dfy, Not([:bus_id, :RouteUsage, :Branded]))
    rename!(dfy_long, :value => :Y)
    dfy_long = @transform(dfy_long, :time = kron(collect(1:20), ones(N)))
    select!(dfy_long, Not(:variable))

    # reshape the odometer variable (Odo1-Odo20)
    dfx = @select(df, :bus_id, :Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7,
                      :Odo8, :Odo9, :Odo10, :Odo11, :Odo12, :Odo13, :Odo14, :Odo15,
                      :Odo16, :Odo17, :Odo18, :Odo19, :Odo20)
    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    rename!(dfx_long, :value => :Odometer)
    dfx_long = @transform(dfx_long, :time = kron(collect(1:20), ones(N)))
    select!(dfx_long, Not(:variable))

    # join the two long data frames
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id, :time])
    sort!(df_long, [:bus_id, :time])

    return df_long
end

################################################################################
# Question 2: static (myopic) logit
################################################################################

"""
    estimate_static_model(df_long)

Binary logit of Y (1 = run, 0 = replace) on Odometer and Branded:
P(Y=1) = Λ(θ0 + θ1·Odometer + θ2·Branded).
"""
function estimate_static_model(df_long)
    return glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink())
end

################################################################################
# Question 3a-3b: wide data and state grids
################################################################################

"""
    load_dynamic_data()

Read busdata.csv and keep it wide. Y, Odo and Xst become N×T matrices; Zst and
Branded are N-vectors. Also builds the grids with create_grids() and returns
everything in one named tuple.
"""
function load_dynamic_data()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS5-ddc/busdata.csv"
    df  = CSV.read(HTTP.get(url).body, DataFrame)

    Y      = Matrix(df[:, r"^Y\d+"])     # choices (N×T): 1 = run, 0 = replace
    X      = Matrix(df[:, r"^Odo\d+"])   # odometer (N×T), 10,000s of miles
    Xstate = Matrix(df[:, r"^Xst\d+"])   # mileage bin (N×T)
    Zstate = Vector(df[:, :Zst])         # route usage bin (N)
    B      = Vector(df[:, :Branded])     # brand dummy (N)
    N, T   = size(Y)

    # 3b: grids and Markov transition matrix (provided code)
    zval, zbin, xval, xbin, xtran = create_grids()

    return (Y = Y, X = X, B = B, Xstate = Xstate, Zstate = Zstate,
            N = N, T = T,
            xval = xval, xbin = xbin, zval = zval, zbin = zbin, xtran = xtran,
            β = 0.9)
end

################################################################################
# Question 3c: future values by backward recursion
################################################################################

#=
    compute_future_value!(FV, θ, d)

FV has size (zbin*xbin, 2, T+1). FV[:, :, T+1] = 0 (terminal condition).
For each grid state row = x + (z-1)*xbin, brand b and period t:
    v1 = θ0 + θ1·xval[x] + θ2·b + xtran[row,:]' · FV[z-block, b+1, t+1]
    v0 =                         xtran[1+(z-1)*xbin,:]' · FV[z-block, b+1, t+1]
    FV[row, b+1, t] = β·log(exp(v0) + exp(v1))
Note: FV already includes β, so β is NOT applied again when FV is used.
=#
@views @inbounds function compute_future_value!(FV, θ, d)
    for t in d.T:-1:1                     # backwards over time
        for b in 0:1                      # brand
            for z in 1:d.zbin             # route usage
                for x in 1:d.xbin         # mileage
                    # row of the transition matrix for this (x, z)
                    row = x + (z-1)*d.xbin
                    # value of running the bus
                    v1 = θ[1] + θ[2]*d.xval[x] + θ[3]*b + d.xtran[row,:]'*FV[(z-1)*d.xbin+1:z*d.xbin, b+1, t+1]
                    # value of replacing the engine (mileage resets to 0)
                    v0 = d.xtran[1+(z-1)*d.xbin,:]'*FV[(z-1)*d.xbin+1:z*d.xbin, b+1, t+1]
                    # future value for period t-1
                    FV[row, b+1, t] = d.β * log(exp(v0) + exp(v1))
                end
            end
        end
    end
    return nothing
end

################################################################################
# Question 3d-3f: log likelihood (Optim-ready, θ is the first argument)
################################################################################

#=
    log_likelihood_dynamic(θ, d)

Returns the NEGATIVE log likelihood. Using only observed states:
    row0 = 1 + (Zst[i]-1)*xbin             (bus replaced)
    row1 = Xst[i,t] + (Zst[i]-1)*xbin      (bus not replaced)
    v1-v0 = θ0 + θ1·Odo[i,t] + θ2·B[i]
            + (xtran[row1,:] - xtran[row0,:])' · FV[row0:row0+xbin-1, B[i]+1, t+1]
    P1 = exp(v1-v0)/(1+exp(v1-v0)),  P0 = 1 - P1                 (7)
    ℓ  = Σ_i Σ_t [Y·log P1 + (1-Y)·log P0]                         (6)
=#
@views @inbounds function log_likelihood_dynamic(θ, d)
    # solve the model for this θ
    FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
    compute_future_value!(FV, θ, d)

    # likelihood over observed states
    loglike = 0.0
    for i in 1:d.N
        row0 = 1 + (d.Zstate[i] - 1) * d.xbin
        for t in 1:d.T
            row1 = d.Xstate[i,t] + (d.Zstate[i] - 1) * d.xbin

            # flow utility part of v1 - v0
            flow_diff = θ[1] + θ[2]*d.X[i,t] + θ[3]*d.B[i]
            # discounted future value part of v1 - v0
            v_diff = flow_diff + (d.xtran[row1,:] .- d.xtran[row0,:])'*FV[row0:row0+d.xbin-1, d.B[i]+1, t+1]

            # choice probabilities, equation (7)
            P1 = exp(v_diff) / (1 + exp(v_diff))
            P0 = 1 - P1

            # log likelihood, equation (6)
            loglike += d.Y[i,t]*log(P1) + (1 - d.Y[i,t])*log(P0)
        end
    end
    return -loglike
end

################################################################################
# Question 3e, 3h: estimate with Optim
################################################################################

"""
    estimate_dynamic_model(d, θ_start)

Minimize the negative log likelihood with Optim, starting from θ_start
(the static estimates from Question 2). Returns the Optim result.
"""
function estimate_dynamic_model(d, θ_start)
    result = optimize(θ -> log_likelihood_dynamic(θ, d), θ_start, LBFGS(),
                      Optim.Options(g_tol = 1e-5))
    println("Dynamic estimates (β = ", d.β, "): ", result.minimizer)
    println("Log likelihood: ", -result.minimum)
    return result
end

################################################################################
# Question 3g: wrap everything in one function
################################################################################

"""
    main()

Runs Questions 1-3. Returns the static GLM fit and the dynamic Optim result.
"""
function main()
    # Question 1
    df_long = load_static_data()
    println(first(df_long, 5))

    # Question 2
    θ̂_static = estimate_static_model(df_long)
    println(θ̂_static)

    # Question 3
    d = load_dynamic_data()
    result_dynamic = estimate_dynamic_model(d, coef(θ̂_static))

    return θ̂_static, result_dynamic
end
