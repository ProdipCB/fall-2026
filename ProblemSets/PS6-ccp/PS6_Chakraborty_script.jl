#=
Problem Set 6 - Script
ECON 6343: Econometrics III
Prodip Chakraborty

How to run (from this folder):   julia PS6_Chakraborty_script.jl
Unit tests:                       julia PS6_Chakraborty_tests.jl

------------------------------------------------------------------------
WRITEUP
------------------------------------------------------------------------
Q1. busdata.csv (second CSV from PS5) is reshaped to long format:
    1,000 buses x 20 periods = 20,000 rows, sorted by bus_id then time.
    Y = 1 means "keep running", Y = 0 means "replace engine".

Q2. Flexible logit of Y on the full interaction of Odometer, Odometer^2,
    RouteUsage, RouteUsage^2, Branded, time, time^2
    (2^7 = 128 coefficients including the intercept).

Q3. CCP estimation with β = 0.9.
    v1 - v0 = θ0 + θ1*Odometer + θ2*Branded
              - β Σ log p0(x', b, t+1) [f1(x'|x) - f0(x'|x)]
    The last term is built from the flexible-logit CCPs and enters the
    final logit as an offset (coefficient fixed at 1).

    Expected results (approximately):
        θ0 (Intercept)  ≈  1.936   (s.e. ≈ 0.033)
        θ1 (Odometer)   ≈ -0.150   (s.e. ≈ 0.006)
        θ2 (Branded)    ≈  1.086   (s.e. ≈ 0.039)
    [Compare with your PS5 backward-recursion estimates here.]
    My own MLE (optional part) gives the same numbers as GLM.

    Implementation notes:
    (1) State data frame: the handout writes RouteUsage = kron(ones(xbin), zval),
        but xtran stacks rows as (z-1)*xbin + x, so the matching vector is
        kron(zval, ones(xbin)). With the handout version the future values do
        not line up with xtran rows: the mapped term FVT1 reaches about 53
        instead of about 3.5, and the estimates move to roughly (2.02, -0.19, 1.14).
    (2) p0 = 1 - predict(), since predict() gives P(Y = 1) = P(run).
    (3) At very high mileage the 7th-order polynomial gives predict() == 1.0,
        so log(p0) = -Inf and FVT1 becomes NaN. I use the exact identity
        -log p0 = log(1 + exp(η)) (softplus of the logit index), which is stable.
    (4) In the flexible logit, Odometer and time enter as Odometer/10 and
        time/10. This is the same model, but without it GLM.jl drops 37 of
        the 128 columns (including the intercept) as "collinear" because the
        raw 7th-order products are huge. The intercept being dropped shows up
        as mean fitted probability != mean(Y).
    (5) FV[:, b+1, t] stores period-t values for t = 2..T; the data in period t
        uses FV[:, :, t+1]; FV[:, :, T+1] = 0 is the terminal period.
------------------------------------------------------------------------
=#

include(joinpath(@__DIR__, "PS6_Chakraborty_source.jl"))

@time main()
