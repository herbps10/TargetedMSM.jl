using TargetedMSM

using LinearAlgebra
using Distributions
using Random
using DataFrames
using DataFramesMeta
using ProgressMeter
using Optim
using CSV
using Plots

logit(p) = log(p / (1 - p))
inv_logit(p) = exp(p) / (exp(p) + 1)

function simulate_data(N, rng; linear = true)
    data = DataFrame(
        p1 = fill(0., N),
        p2 = fill(0., N),
        X1 = fill(0., N),
        X2 = fill(0., N),
        X3  = fill(0., N),
        X4  = fill(0., N),
        A  = fill(0, N),
        Y0 = fill(0.0, N),
        Y1 = fill(0.0, N),
        Y  = fill(0.0, N),
        diff = fill(0.0, N)
    )

    data[:, :X1] = rand(rng, Normal(0, 1), N)
    data[:, :X2] = rand(rng, Normal(0, 1), N)
    data[:, :X3] = rand(rng, Normal(0, 1), N)
    data[:, :X4] = rand(rng, Normal(0, 1), N)

    for i = 1:N
        p1 = 0.5 * data[i, :X1] - 0.5 * data[i, :X2] + 0.2 * data[i, :X3] - 0.1 * data[i, :X4]
        p2 = data[i, :X2] + data[i, :X3]

        data[i, :p1] = p1
        data[i, :p2] = p2

        data[i, :A]  = rand(rng, Binomial(1, inv_logit(p1)), 1)[1]

        if linear == true
            data[i, :Y0] = rand(rng, Normal(p2, 0.1))
            data[i, :Y1] = rand(rng, Normal(p2 + 3.0 + 1.5 * data[i, :X4], 0.1))

            data[i, :Y]  = data[i, :A] == 1 ? data[i, :Y1] : data[i, :Y0]
        else
            data[i, :Y0] = rand(rng, Bernoulli(inv_logit(p2)))
            data[i, :Y1] = rand(rng, Bernoulli(inv_logit(0.1 * p2 + 0.5 + 0.5 * data[i, :X4])))

            data[i, :Y]  = data[i, :A] == 1 ? data[i, :Y1] : data[i, :Y0]

        end

        data[i, :diff] = data[i, :Y1] - data[i, :Y0]
    end

    return data
end

function linear_working_model(beta::Vector, X::DataFrameRow)
    return first([1 X[:X4]] * beta)
end

function squared_error_loss(t, m)
    return (t - m)^2
end

function dL(t, beta, X)
    -2 * (t - linear_working_model(beta, X)) .* [1 X[:X4]]'
end

function dL_dβ(t, beta, X)
   2 * [1 X[:X4]]' * [1 X[:X4]] 
end

rng = MersenneTwister(1234)

large_data = simulate_data(100000, rng, linear = true)
optim = optimize(beta -> sum((large_data[:, :diff] .- beta[1] - beta[2] .* large_data[:, :X4]).^2), repeat([0.0], 2), LBFGS(), autodiff = :forward)
beta0 = Optim.minimizer(optim)

# TMLE simulation study
#g_correct = [true, false]
#Q_correct = [true, false]
g_correct = [true];
Q_correct = [true];
Ns = [100, 200, 500, 1000];
num_simulations = 1;

simulations = DataFrame(
    collect(Iterators.product(g_correct, Q_correct, Ns, 1:num_simulations))
);

rename!(simulations, ["g_correct", "Q_correct", "N", "index"])
simulations[:, :result] = Vector{Any}(nothing, size(simulations, 1))
simulations[:, :beta1_covered] = fill(false, size(simulations, 1))
simulations[:, :beta2_covered] = fill(false, size(simulations, 1))
simulations[:, :beta1_bias] = fill(0.0, size(simulations, 1))
simulations[:, :beta2_bias] = fill(0.0, size(simulations, 1))

simulations[:, :bayes_beta1_covered] = fill(false, size(simulations, 1))
simulations[:, :bayes_beta2_covered] = fill(false, size(simulations, 1))
simulations[:, :bayes_beta1_bias] = fill(0.0, size(simulations, 1))
simulations[:, :bayes_beta2_bias] = fill(0.0, size(simulations, 1))

bayes = true

println("Starting simulations...")
progress = Progress(size(simulations, 1));
Threads.@threads for index = 1:size(simulations, 1)
    data = simulate_data(simulations[index, :N], rng)

    g_formula = @formula(A ~ X1 + X2 + X3 + X4)
    Q_formula = @formula(Y ~ X1 + X2 + X3 + X4 + A + X1*A + X2*A + X3*A + X4*A)

    g_formula_misspec = @formula(A ~ X1 + X4)
    Q_formula_misspec = @formula(Y ~ X1 + X4 + A + X1*A + X4*A)

    result = TargetedMSM.treatment_effect_modification(
        data[:, :Y], 
        data[:, :A], 
        data[:, [:X1, :X2, :X3, :X4]], 
        (simulations[index, :g_correct] == true ? g_formula : g_formula_misspec),
        (simulations[index, :Q_correct] == true ? Q_formula : Q_formula_misspec),
        @formula(Y2 ~ X1 + X2 + X3 + X4 + A + X1*A + X2*A + X3*A + X4*A),
        2, 
        squared_error_loss,
        linear_working_model;
        seed = 1234,
        proposal_sd = nothing,
        iterations = 10_000,
        bayes = bayes,
        include_prior = true,
        prior = Normal(0, 1),
        dL = dL,
        dL_dβ = dL_dβ
    )

    simulations[index, :result] = result
    simulations[index, [:beta1_covered, :beta2_covered]] = (
        (result[:beta_lower] .<= beta0) .& (result[:beta_upper] .>= beta0)
    )

    simulations[index, [:beta1_bias, :beta2_bias]] = result[:beta] - beta0

    if bayes
        simulations[index, [:bayes_beta1_covered, :bayes_beta2_covered]] = (
            (result[:beta_lower_bayes] .<= beta0) .& (result[:beta_upper_bayes] .>= beta0)
        )
        simulations[index, [:bayes_beta1_bias, :bayes_beta2_bias]] = result[:beta_bayes] - beta0
    end

    next!(progress)
end

results = @combine(groupby(simulations, [:g_correct, :Q_correct, :N]),
    :bias_beta1     = mean(:beta1_bias), 
    :bias_beta2     = mean(:beta2_bias),
    :mse_beta1      = mean(:beta1_bias.^2), 
    :mse_beta2      = mean(:beta2_bias.^2),
    :coverage_beta1 = mean(:beta1_covered),
    :coverage_beta2 = mean(:beta2_covered),
    
    :bayes_bias_beta1     = mean(:bayes_beta1_bias), 
    :bayes_bias_beta2     = mean(:bayes_beta2_bias),
    :bayes_mse_beta1      = mean(:bayes_beta1_bias.^2), 
    :bayes_mse_beta2      = mean(:bayes_beta2_bias.^2),
    :bayes_coverage_beta1 = mean(:bayes_beta1_covered),
    :bayes_coverage_beta2 = mean(:bayes_beta2_covered),
)

CSV.write("results.csv", results)

fig = scatter(
    results[:, :N], results[:, :coverage_beta2], 
    ylim = [0.8, 1],
    label = L"\beta_1", 
    xlab = "N", 
    ylab = "Empirical 95% CI Coverage"
)

scatter!(results[:, :N], results[:, :coverage_beta1], ylim = [0.8, 1], label = L"\beta_2")
plot!([0, 1000], [0.95, 0.95], label = "", color = "black", line = (:dash))

savefig(fig, "coverage.pdf")

println(results)
