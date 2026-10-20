#=
Problem Set 6 - Source code (functions only)
ECON 6343: Econometrics III
Prodip Chakraborty

Rust (1987) bus engine replacement model estimated with
conditional choice probabilities (CCPs).

This file only DEFINES functions. Run PS6_Chakraborty_script.jl to estimate
the model, and PS6_Chakraborty_tests.jl to run the unit tests.
=#

using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV

const SM = GLM.StatsModels   # for formula() and modelcols()

const BUS_URL = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS5-ddc/busdata.csv"


#========================================================================
Grids and state transitions (same code as PS5 / create_grids.jl)
========================================================================#

"""
    xgrid(theta, xval)

Mileage transition matrix for one route-usage value `theta`.
Row = current mileage bin, column = next mileage bin.
"""
function xgrid(theta, xval)
    N       = length(xval)
    xub     = vcat(xval[2:N], Inf)
    xtran1  = zeros(N, N)
    xtran1c = zeros(N, N)
    lcdf    = zeros(N)
    for i = 1:length(xval)
        xtran1[:, i]   = (xub[i] .>= xval) .* (1 .- exp.(-theta * (xub[i] .- xval)) .- lcdf)
        lcdf         .+= xtran1[:, i]
        xtran1c[:, i] .+= lcdf
    end
    return xtran1, xtran1c
end

"""
    create_grids()

Returns `zval, zbin, xval, xbin, xtran`.
`xtran` is (zbin*xbin) x xbin. Rows are stacked in blocks by route usage:
row = (z-1)*xbin + x  (mileage changes fastest inside each z block).
"""
function create_grids()
    zval   = collect([.25:.01:1.25]...)
    zbin   = length(zval)
    xval   = collect([0:.125:25]...)
    xbin   = length(xval)
    tbin   = xbin * zbin
    xtran  = zeros(tbin, xbin)
    xtranc = zeros(xbin, xbin, xbin)
    for z = 1:zbin
        xtran[(z-1)*xbin+1:z*xbin, :], xtranc[:, :, z] = xgrid(zval[z], xval)
    end
    return zval, zbin, xval, xbin, xtran
end


#========================================================================
Question 1: read data and reshape to long
========================================================================#

"""
    load_and_reshape_data(url = BUS_URL)

Reads the PS5 dynamic bus data (busdata.csv) and returns `df_long`,
one row per bus-period, sorted by `bus_id` then `time`.
Columns: bus_id, time, Y, Odometer, Xstate, RouteUsage, Branded, Zst.
Y = 1 means the bus keeps running; Y = 0 means the engine is replaced.
"""
function load_and_reshape_data(url::String = BUS_URL)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    return reshape_to_long(df)
end

"""
    reshape_to_long(df)

Wide -> long. Kept separate from the download so it can be unit tested.
"""
function reshape_to_long(df::DataFrame)
    N = nrow(df)
    T = count(n -> occursin(r"^Y\d+$", n), names(df))
    pieces = [DataFrame(bus_id     = 1:N,
                        time       = fill(t, N),
                        Y          = df[!, "Y$t"],
                        Odometer   = df[!, "Odo$t"],
                        Xstate     = df[!, "Xst$t"],
                        RouteUsage = df[!, "RouteUsage"],
                        Branded    = df[!, "Branded"],
                        Zst        = df[!, "Zst"]) for t in 1:T]
    df_long = vcat(pieces...)
    sort!(df_long, [:bus_id, :time])
    return df_long
end

"""
    get_state_arrays(df_long)

From the long data (sorted by bus_id, time) returns
`Xstate` (N x T mileage-state matrix), `Zstate` (N vector), `B` (N vector of Branded).
"""
function get_state_arrays(df_long::DataFrame)
    T = maximum(df_long.time)
    N = length(unique(df_long.bus_id))
    @assert nrow(df_long) == N * T
    Xstate = permutedims(reshape(df_long.Xstate, T, N))   # N x T
    Zstate = df_long.Zst[1:T:end]
    B      = df_long.Branded[1:T:end]
    return Xstate, Zstate, B
end


#========================================================================
Question 2: flexible logit
========================================================================#

"""
    add_squared_terms(df)

Adds the variables used in the flexible logit:
- Odometer_sq, RouteUsage_sq, time_sq (squares in original units), and
- Odo10 = Odometer/10, Odo10_sq, time10 = time/10, time10_sq (rescaled versions).

Why the rescaled versions: with raw units, the 7th-order products are huge
(e.g. Odometer^2 * time^2 is up to 625*400), so the design matrix is badly
conditioned (condition number about 1e10). GLM.jl's collinearity check
(pivoted Cholesky, dropcollinear=true) then drops 37 of the 128 columns,
including the intercept, and the fit is wrong: mean fitted probability is not
equal to mean(Y). Dividing Odometer and time by 10 gives the SAME model (same
column space, same likelihood, same fitted probabilities) but a condition
number of about 3e6, and no columns are dropped.
"""
function add_squared_terms(df::DataFrame)
    return @transform(df,
                      :Odometer_sq   = :Odometer .^ 2,
                      :RouteUsage_sq = :RouteUsage .^ 2,
                      :time_sq       = :time .^ 2,
                      :Odo10         = :Odometer ./ 10,
                      :Odo10_sq      = (:Odometer ./ 10) .^ 2,
                      :time10        = :time ./ 10,
                      :time10_sq     = (:time ./ 10) .^ 2)
end

"""
    estimate_flexible_logit(df)

Logit of Y on a fully interacted set of Odometer, Odometer^2, RouteUsage,
RouteUsage^2, Branded, time, time^2 (all terms of 1st to 7th order = 128 coefficients
including the intercept). Odometer and time enter as Odometer/10 and time/10
for numerical stability only (see add_squared_terms); the model is unchanged.
"""
function estimate_flexible_logit(df::DataFrame)
    df = add_squared_terms(df)
    flex_logit = glm(@formula(Y ~ Odo10 * Odo10_sq * RouteUsage * RouteUsage_sq *
                                  Branded * time10 * time10_sq),
                     df, Binomial(), LogitLink())
    return flex_logit
end


#========================================================================
Question 3: CCP estimation
========================================================================#

"""
    construct_state_space(xbin, zbin, xval, zval)

One row per grid point, in the SAME row order as `xtran`.

NOTE: the PS handout writes RouteUsage = kron(ones(xbin), zval). That cycles z
fastest, which does not match xtran (x cycles fastest within each z block).
Row r of xtran has x = xval[mod1(r, xbin)] and z = zval[cld(r, xbin)], so the
matching vector is kron(zval, ones(xbin)). Odometer = kron(ones(zbin), xval) as in the PS.
"""
function construct_state_space(xbin::Int, zbin::Int, xval::Vector, zval::Vector)
    n = xbin * zbin
    state_df = DataFrame(Odometer   = kron(ones(zbin), xval),
                         RouteUsage = kron(zval, ones(xbin)),
                         Branded    = zeros(Int, n),
                         time       = zeros(Int, n))
    return state_df
end

"""
    softplus(x) = log(1 + exp(x)), computed without overflow.
"""
softplus(x::Real) = x > zero(x) ? x + log1p(exp(-x)) : log1p(exp(x))

"""
    linear_predictor(flex_logit, df)

Returns the logit index η = X*b for new data `df`, using the fitted formula.
P(run) = 1/(1+exp(-η)) and p0 = P(replace) = 1/(1+exp(η)).
"""
function linear_predictor(flex_logit, df::DataFrame)
    Xnew = SM.modelcols(SM.formula(flex_logit).rhs, df)
    return Xnew * coef(flex_logit)
end

"""
    compute_future_values(state_df, flex_logit, xtran, xbin, zbin, T, β)

FV[:, b+1, t] = -β * log p0(x, z, b, t) for t = 2..T. FV[:,:,1] and FV[:,:,T+1] stay 0
(T+1 is the terminal period). The data in period t uses FV[:, :, t+1].

p0 = 1 - predict() (Y = 1 is "run"). For very high mileage the polynomial gives
predict() == 1.0 exactly, so log(1 - predict()) = -Inf and 0*Inf = NaN later.
I therefore use the identity -log p0 = log(1 + exp(η)) = softplus(η),
which is the same number but never overflows.
"""
function compute_future_values(state_df::DataFrame,
                               flex_logit,
                               xtran::Matrix,
                               xbin::Int,
                               zbin::Int,
                               T::Int,
                               β::Float64)
    @assert nrow(state_df) == size(xtran, 1) == xbin * zbin
    FV  = zeros(size(xtran, 1), 2, T + 1)
    sdf = copy(state_df)
    for t in 2:T
        for b in 0:1
            sdf.time    .= t
            sdf.Branded .= b
            η = linear_predictor(flex_logit, add_squared_terms(sdf))
            FV[:, b+1, t] = β .* softplus.(η)        # = -β log p0
        end
    end
    return FV
end

"""
    compute_fvt1(df_long, FV, xtran, Xstate, Zstate, B, xbin)

Maps the future values to the data:
FVT1[i,t] = (xtran[row1,:] - xtran[row0,:])' * FV[row0:row0+xbin-1, B[i]+1, t+1]
row0 = mileage-zero row for bus i's route usage (state after replacement),
row1 = bus i's current mileage row. Returns a vector in long order (bus, then time).
"""
function compute_fvt1(df_long::DataFrame,
                      FV::Array{Float64,3},
                      xtran::Matrix,
                      Xstate::Matrix,
                      Zstate::Vector,
                      B::Vector,
                      xbin::Int)
    N, T = size(Xstate)
    @assert nrow(df_long) == N * T
    FVT1 = zeros(N, T)
    for i in 1:N
        row0 = (Zstate[i] - 1) * xbin + 1
        for t in 1:T
            row1 = Xstate[i, t] + (Zstate[i] - 1) * xbin
            FVT1[i, t] = @views (xtran[row1, :] .- xtran[row0, :])' *
                                FV[row0:row0+xbin-1, B[i]+1, t+1]
        end
    end
    return FVT1'[:]      # row i, then t  ->  matches df_long sorted by bus_id, time
end

"""
    estimate_structural_params(df_long, fvt1)

Logit of Y on Odometer and Branded with the future value as an offset
(coefficient fixed at 1).
"""
function estimate_structural_params(df_long::DataFrame, fvt1::Vector)
    @assert length(fvt1) == nrow(df_long)
    df_long = copy(df_long)
    df_long.fv = fvt1
    theta_hat = glm(@formula(Y ~ Odometer + Branded),
                    df_long, Binomial(), LogitLink(),
                    offset = df_long.fv)
    return theta_hat
end

"""
    logit_offset_mle(y, X, offset)

(Optional part) My own binary logit with an offset whose coefficient is 1.
Returns (θ, standard errors).
"""
function logit_offset_mle(y::AbstractVector, X::AbstractMatrix, offset::AbstractVector)
    function nll(θ)
        η = X * θ .+ offset
        return -sum(y .* η .- softplus.(η))
    end
    function g!(G, θ)
        η = X * θ .+ offset
        p = 1 ./ (1 .+ exp.(-η))
        G .= -(X' * (y .- p))
        return G
    end
    res = optimize(nll, g!, zeros(size(X, 2)), LBFGS(), Optim.Options(g_tol = 1e-9))
    θ   = Optim.minimizer(res)
    η   = X * θ .+ offset
    p   = 1 ./ (1 .+ exp.(-η))
    H   = X' * (X .* (p .* (1 .- p)))
    se  = sqrt.(diag(inv(H)))
    return θ, se
end


#========================================================================
Wrapper
========================================================================#

"""
    main()

Runs everything: Q1 -> Q2 -> Q3. Returns the structural GLM.
"""
function main()
    println("="^60)
    println("Rust (1987) Bus Engine Replacement Model - CCP Estimation")
    println("="^60)
    β = 0.9

    # Q1
    println("\nQ1: Loading and reshaping data...")
    df_long = load_and_reshape_data(BUS_URL)
    Xstate, Zstate, B = get_state_arrays(df_long)
    T = size(Xstate, 2)

    # Q2
    println("\nQ2: Estimating flexible logit...")
    flex_logit = estimate_flexible_logit(df_long)
    println("Number of coefficients: ", length(coef(flex_logit)))

    # Q3a
    println("\nQ3a: Constructing state transition matrices...")
    zval, zbin, xval, xbin, xtran = create_grids()

    # Q3b
    println("\nQ3b: Computing future values...")
    state_df = construct_state_space(xbin, zbin, xval, zval)
    FV       = compute_future_values(state_df, flex_logit, xtran, xbin, zbin, T, β)
    fvt1     = compute_fvt1(df_long, FV, xtran, Xstate, Zstate, B, xbin)

    # Q3c
    println("\nQ3c: Estimating structural parameters...")
    theta_hat_ccp_glm = estimate_structural_params(df_long, fvt1)

    println("\n" * "="^60)
    println("RESULTS: structural parameters (GLM with offset)")
    println("="^60)
    println(coeftable(theta_hat_ccp_glm))

    # Q3d (optional): own MLE with the offset restricted to 1
    X = hcat(ones(nrow(df_long)), df_long.Odometer, df_long.Branded)
    θ_own, se_own = logit_offset_mle(df_long.Y, X, fvt1)
    println("\nOwn MLE (θ0, θ1, θ2): ", round.(θ_own, digits = 4))
    println("Own MLE std. errors : ", round.(se_own, digits = 4))

    return theta_hat_ccp_glm
end
