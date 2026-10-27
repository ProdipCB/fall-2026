################################################################################
# Problem Set 7 - Script (runs everything)
# ECON 6343: Econometrics III
# Prodip Chakraborty
################################################################################

include("PS7_Chakraborty_source.jl")

function main()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df, X_wage, y_wage = load_data(url)
    df, X, y = prepare_occupation_data(df)
    K = size(X, 2)
    J = 7

    #--------------------------------------------------------------------------
    # Question 1: OLS by GMM
    #--------------------------------------------------------------------------
    println("\n", "="^70, "\nQuestion 1: OLS by GMM\n", "="^70)
    Random.seed!(1234)
    β_gmm = optimize(b -> ols_gmm(b, X_wage, y_wage), rand(size(X_wage, 2)),
                     LBFGS(), Optim.Options(g_tol = 1e-8, iterations = 100_000))
    β_ols = ols_closed_form(X_wage, y_wage)
    println("GMM estimates:         ", round.(β_gmm.minimizer, digits = 5))
    println("Closed-form OLS:       ", round.(β_ols, digits = 5))
    println("Max abs difference:    ", maximum(abs.(β_gmm.minimizer .- β_ols)))
    # Answer: With g = y - Xβ and W = I, J(β) = (y-Xβ)'(y-Xβ). This is the
    # sum of squared residuals, so the GMM estimate is exactly OLS.

    #--------------------------------------------------------------------------
    # Question 2: Multinomial logit
    #--------------------------------------------------------------------------
    println("\n", "="^70, "\nQuestion 2: Multinomial logit\n", "="^70)

    # (a) MLE
    svals = mlogit_starting_values(df, X; J = J)
    α_mle = optimize(a -> mlogit_mle(a, X, y), svals, LBFGS(),
                     Optim.Options(g_tol = 1e-5, iterations = 100_000))
    println("(a) MLE: -loglik = ", round(α_mle.minimum, digits = 4))
    show_coefs(α_mle.minimizer, K, J)

    # (b) GMM, MLE estimates as starting values
    α_gmm_b = optimize(a -> mlogit_gmm_overid(a, X, y), α_mle.minimizer, LBFGS(),
                       Optim.Options(g_tol = 1e-5, iterations = 100_000))
    println("\n(b) GMM from MLE start: objective = ", round(α_gmm_b.minimum, digits = 4),
            ",  converged = ", Optim.converged(α_gmm_b))
    show_coefs(α_gmm_b.minimizer, K, J)
    println("  max |α_GMM(b) - α_MLE| = ",
            round(maximum(abs.(α_gmm_b.minimizer .- α_mle.minimizer)), digits = 4))
    flush(stdout)

    # (c) GMM, random starting values (several draws to check)
    println("\n(c) GMM from random starts:")
    results_c = []
    for s in 1:3
        Random.seed!(s)
        a0 = rand(length(svals))
        r = optimize(a -> mlogit_gmm_overid(a, X, y), a0, LBFGS(),
                     Optim.Options(g_tol = 1e-5, iterations = 100_000))
        push!(results_c, r)
        println("\n  seed $s: objective = ", round(r.minimum, digits = 4),
                ",  converged = ", Optim.converged(r),
                ",  iterations = ", r.iterations,
                ",  max |α_c - α_b| = ",
                round(maximum(abs.(r.minimizer .- α_gmm_b.minimizer)), digits = 4))
        show_coefs(r.minimizer, K, J)
        flush(stdout)
    end

    # Answers to Q2 (numbers are from my run; see PS7_Chakraborty_output.txt):
    # - (a) MLE converges; -log likelihood = 3605.09.
    # - (b) GMM from the MLE start converges (objective about 1699.82).
    #   The estimates are similar to the MLE in sign and size for most
    #   coefficients, but not equal: the largest gap is about 0.9 (collgrad,
    #   occupation 6). This is expected. With W = I the moments d - P give a
    #   nonlinear least squares problem, which is a different estimator from
    #   MLE, so the two answers need not be the same.
    # - (c) From random starts the answers are different from (b):
    #     * seed 1 stops at a HIGHER objective (about 1701.4) with very
    #       different occupation-5 coefficients (age about -0.65). This is a
    #       local minimum.
    #     * seed 2 jumps after one step into a flat region (huge
    #       coefficients, e.g. age about 456, objective 3840). The
    #       probabilities are all 0 or 1 there, the gradient is zero, and
    #       Optim reports "converged" even though this is not a minimum.
    #     * seed 3 reaches the same point as (b) (objective 1699.82, max
    #       difference 0.001), but only after about 12,000 iterations.
    # - Is the objective globally concave? No. (Maximizing -J(α) is not
    #   globally concave; equivalently J(α) is not globally convex.)
    #   Different starting values give different end points, with
    #   different objective values, and there are flat regions. So the
    #   GMM answer depends on the starting values. The logit log likelihood
    #   IS globally concave, so the MLE is a safe starting point for GMM.

    #--------------------------------------------------------------------------
    # Question 3: Simulate and recover
    #--------------------------------------------------------------------------
    flush(stdout)
    println("\n", "="^70, "\nQuestion 3: Simulate data and recover β\n", "="^70)
    Nsim, Jsim = 100_000, 4
    β_true = default_beta()
    Ksim = size(β_true, 1)
    true_vec = vec(β_true[:, 1:Jsim-1])

    # Method (a)-(e): uniform shocks
    XS, YS = sim_logit(Nsim, Jsim, β_true; seed = 1234)
    println("Choice shares (uniform method): ",
            round.([mean(YS .== j) for j in 1:Jsim], digits = 3))
    α_sim = optimize(a -> mlogit_mle(a, XS, YS), zeros(Ksim * (Jsim - 1)), LBFGS(),
                     Optim.Options(g_tol = 1e-6, iterations = 100_000))
    println("True β (alternatives 1..J-1, stacked):  ", true_vec)
    println("Estimated β (uniform method):           ", round.(α_sim.minimizer, digits = 3))

    # Method (f): T1EV / Gumbel shocks
    XG, YG = sim_logit_w_gumbel(Nsim, Jsim, β_true; seed = 4321)
    α_simG = optimize(a -> mlogit_mle(a, XG, YG), zeros(Ksim * (Jsim - 1)), LBFGS(),
                      Optim.Options(g_tol = 1e-6, iterations = 100_000))
    println("Estimated β (Gumbel method):            ", round.(α_simG.minimizer, digits = 3))
    # Answer: with N = 100,000, both methods give estimates within about
    # 0.05 of the true values (largest gap 0.045), so the simulation code
    # works and the MLE recovers β.

    #--------------------------------------------------------------------------
    # Question 5: Multinomial logit by SMM
    #--------------------------------------------------------------------------
    flush(stdout)
    println("\n", "="^70, "\nQuestion 5: Multinomial logit by SMM\n", "="^70)
    # Simulated choices come from the Q3(f) Gumbel code.
    # The shocks are drawn once and held fixed (common random numbers).
    # The simulated frequencies are step functions of α, so the objective
    # has zero gradient almost everywhere. A gradient method (LBFGS) would
    # stop at the starting values, so I use the derivative-free Nelder-Mead.
    D = 100
    shocks = draw_gumbel_shocks(length(y), J, D; seed = 1234)
    α_smm = optimize(a -> mlogit_smm_overid(a, X, y, shocks), α_mle.minimizer,
                     NelderMead(), Optim.Options(iterations = 3_000, g_tol = 1e-8))
    println("SMM objective at MLE start: ",
            round(mlogit_smm_overid(α_mle.minimizer, X, y, shocks), digits = 4))
    println("SMM objective at optimum:   ", round(α_smm.minimum, digits = 4))
    println("SMM estimates (D = $D):")
    show_coefs(α_smm.minimizer, K, J)
    println("\nMax |SMM - MLE| = ",
            round(maximum(abs.(α_smm.minimizer .- α_mle.minimizer)), digits = 4))

    # Check SMM on simulated data where the truth is known (uses Q3 code)
    XS2, YS2 = sim_logit_w_gumbel(5_000, Jsim, β_true; seed = 99)
    shocks2 = draw_gumbel_shocks(5_000, Jsim, D; seed = 2024)
    α_smm_sim = optimize(a -> mlogit_smm_overid(a, XS2, YS2, shocks2),
                         zeros(Ksim * (Jsim - 1)), NelderMead(),
                         Optim.Options(iterations = 3_000))
    println("\nSMM on simulated data (N = 5,000):")
    println("  True β:      ", true_vec)
    println("  SMM estimate:", round.(α_smm_sim.minimizer, digits = 3))
    # Answers (numbers from my run):
    # - Real data: SMM lowers the objective from 1714.47 (at the MLE) to
    #   1710.84. The SMM estimates stay close to the MLE (largest gap about
    #   0.11) and have the same signs. Part of this closeness is because
    #   Nelder-Mead starts at the MLE.
    # - Simulated data (N = 5,000, start at zeros): most SMM estimates are
    #   near the true β (e.g. 1.02 vs 1.0, -0.50 vs -0.5), but some are off
    #   (0.09 vs 0.3, -2.39 vs -2.0). SMM is noisier than MLE because of
    #   simulation error (finite D = 100), the smaller N, and the
    #   non-smooth objective, which Nelder-Mead (3,000 iterations) handles
    #   slowly in 9 dimensions. More draws (D) and more iterations help.

    return (β_gmm = β_gmm, α_mle = α_mle, α_gmm_b = α_gmm_b,
            α_gmm_c = results_c, α_sim = α_sim, α_simG = α_simG, α_smm = α_smm)
end

show_coefs(α, K, J) = println(coef_table(α, K, J))

@time main()
