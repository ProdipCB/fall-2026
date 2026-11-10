#==================================================================================
Problem Set 8 - Main Script
ECON 6343: Econometrics III
Author: Prodip Chakraborty
==================================================================================#

cd(@__DIR__)
include("PS8_Chakraborty_source.jl")

"""
    main()

Run Questions 1-6 of Problem Set 8.
"""
function main()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS8-factor/nlsy.csv"
    src = isfile("nlsy.csv") ? "nlsy.csv" : url      # use a local copy if there is one

    println("="^80)
    println("Problem Set 8: Factor Models and Dimension Reduction")
    println("="^80)

    println("\nLoading data...")
    df = load_data(src)
    println("Data loaded successfully. Dimensions: ", size(df))

    #------------------------------------------------------------------------------
    # Question 1
    #------------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 1: Base Regression (without ASVAB)")
    println("="^80)
    est = estimate_base_regression(df)
    display(coeftable(est))

    #------------------------------------------------------------------------------
    # Question 2
    #------------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 2: ASVAB Correlations")
    println("="^80)
    cordf = compute_asvab_correlations(df)
    show(stdout, MIME("text/plain"), round.(cordf, digits = 3); allrows = true, allcols = true)
    println()

    #------------------------------------------------------------------------------
    # Question 3
    #------------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 3: Full Regression (with all ASVAB)")
    println("="^80)
    estfull = estimate_full_regression(df)
    display(coeftable(estfull))

    #------------------------------------------------------------------------------
    # Question 4
    #------------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 4: PCA Regression")
    println("="^80)
    estpca = estimate_pca_regression(df)
    display(coeftable(estpca))

    #------------------------------------------------------------------------------
    # Question 5
    #------------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 5: Factor Analysis Regression")
    println("="^80)
    estfac = estimate_factor_regression(df)
    display(coeftable(estfac))

    #------------------------------------------------------------------------------
    # Question 6
    #------------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 6: Full Factor Model (MLE)")
    println("="^80)
    X, y, Xfac, asvabs = prepare_factor_matrices(df)
    start_vals = get_starting_values(X, y, Xfac, asvabs)
    println("Number of parameters: ", length(start_vals))

    println("\nEstimating full factor model (R = 41 quadrature points)...")
    θ̂, se, loglike = run_estimation(df, start_vals; R = 41, show_trace = true)

    results = format_results(θ̂, se, loglike, asvabs)
    results.coefficient = round.(results.coefficient, digits = 4)
    results.std_error   = round.(results.std_error,   digits = 4)
    results.z_stat      = round.(results.z_stat,      digits = 2)
    show(stdout, MIME("text/plain"), results; allrows = true, allcols = true)
    println("\nLog likelihood: ", round(loglike, digits = 2))

    CSV.write("PS8_Chakraborty_factor_results.csv", results)
    println("Saved results to PS8_Chakraborty_factor_results.csv")

    println("\n" * "="^80)
    println("Analysis complete!")
    println("="^80)
    return nothing
end

main()

#==================================================================================
WRITEUP (N = 2,438)

Q1. Base regression
    black -0.167 (se 0.024), hispanic -0.054 (0.026), female -0.155 (0.020),
    schoolt 0.005 (0.005), gradHS 0.196 (0.049), grad4yr 0.299 (0.028).
    I see a large wage gap for black workers and women. A 4-year degree is linked
    to about 30 log points higher wages.

Q2. ASVAB correlations
    All correlations are positive and between 0.42 and 0.80. The highest is
    AR-MK (0.80), then PC-WK (0.75), MK-PC (0.72) and AR-PC (0.71).
    The six tests are highly correlated. They seem to measure one common
    ability.

Q3. Regression with all six ASVABs
    Yes, I think it is a problem. Because the tests are so correlated, there is
    multicollinearity. Only AR (0.045) and MK (0.080) are significant. PC even
    gets a negative sign (-0.022). The separate test coefficients are not
    precise, and I cannot read them one by one. Also, the black gap falls to
    -0.088, the hispanic gap goes to about 0, and grad4yr falls to 0.235. So
    part of the Q1 gaps and the schooling effect is really ability.

Q4. PCA regression
    The first principal component explains about 67% of the ASVAB variance. All
    six loadings have the same sign, so it is a general ability index.
    asvabPCA = 0.053 (se 0.006), and it is very significant. The other
    coefficients are close to Q3 (black -0.094, hispanic -0.002, grad4yr 0.242),
    but now I use one regressor instead of six. (I set the sign so that a
    higher PC means higher scores, because the sign of a PC is arbitrary.)

Q5. Factor Analysis regression
    asvabFactor = 0.115 (se 0.013), and it is very significant. The other
    coefficients are almost the same as with PCA (black -0.086, grad4yr 0.239).
    The factor coefficient is bigger than the PCA one only because of scale:
    the PC has SD about 2.0 and the factor has SD about 0.96. Per one SD,
    both give about 0.11 higher log wage.

Q6. Full factor model by MLE (Gauss-Legendre quadrature, R = 41, on [-5,5])
    Log likelihood = -17,524.0. There are 45 parameters.
    Wage equation:
      black -0.187 (0.025), hispanic -0.081 (0.026), female -0.135 (0.020),
      schoolt -0.003 (0.005), gradHS 0.107 (0.050), grad4yr 0.239 (0.028),
      constant 2.155 (0.051)
      delta (factor loading in wage) = 0.105 (0.012)
      sigma_wage = 0.468
    Measurement loadings (gamma):
      AR 0.771, CS 0.551, MK 0.816, NO 0.643, PC 0.740, WK 0.662 (se ~0.015-0.019)
    Measurement sigmas:
      AR 0.480, CS 0.783, MK 0.436, NO 0.729, PC 0.558, WK 0.612

    The latent ability raises all six test scores and also raises wages
    (delta = 0.105, so one SD more ability gives about 10.5% higher wages).
    Once I control for ability, grad4yr is about 0.24 instead of 0.30 in Q1.
    So some of the raw college premium is ability bias. Unlike Q3-Q5, the black
    and hispanic gaps here are close to the Q1 values. The reason is that the
    measurement equations take out race and sex differences in test scores
    before they define the factor. In Q4-Q5 the index still includes those
    group differences.

    Note on quadrature points: I first used R = 9 (as in PS4). The log
    likelihood was -17,909.5, and the loadings were much smaller (about 0.59
    for AR). With R = 25, 41, 61 and 101 the log likelihood is -17,524.7,
    -17,524.0, -17,524.0 and -17,524.0. So R = 9 is too few nodes here, and
    I use R = 41.

Q7. Unit tests are in PS8_Chakraborty_tests.jl.
==================================================================================#
