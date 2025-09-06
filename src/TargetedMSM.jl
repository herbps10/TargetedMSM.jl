module TargetedMSM

using LinearAlgebra
using DataFrames
using ForwardDiff
using GLM
using Statistics
using Distributions
using Random
using NLsolve
using Optim
using StatsFuns

"""
    tmle(data, p::Integer, L, m, ℒ)

    m(beta::Vector, X::DataFrameRow)
    L(t, Z::DataFrameRow)
"""
function treatment_effect_modification(
    Y, 
    A, 
    X, 
    g_formula, 
    Qbar_formula, 
    p, 
    L::Function, 
    m::Function;
    Qbar = nothing,
    Qbar0 = nothing,
    Qbar1 = nothing,
    g = nothing,
    var_formula = nothing, 
    linear = true,
    bayes = false,
    iterations::Int = 10_000, 
    proposal_sd = nothing,
    seed = 1, 
    include_prior = true, 
    prior = Normal(0, 1),
    dL = nothing,
    dL_dβ = nothing,
    range_threshold = 0.1,
)

    rng = MersenneTwister(seed)
    Lm = (t, beta::Vector, X) -> L(t, m(beta, X))

    # Derivatives of loss function

    if dL === nothing
        dL = (t, beta, X) -> ForwardDiff.gradient(beta -> Lm(t, beta, X), beta)
    end

    ddL(t, beta, X) = ForwardDiff.hessian(beta -> Lm(t, beta, X), beta)
    ∇dL(t, beta, X) = ForwardDiff.jacobian(t -> dL(first(t), beta, X), [t])
    
    if dL_dβ === nothing
        dL_dβ_config = ForwardDiff.JacobianConfig(nothing, repeat([0.0], p))
        dL_dβ = (t, beta, X) -> ForwardDiff.jacobian(β -> dL(t, β, X), beta, dL_dβ_config)
    end

    # Number of observations
    n = size(Y, 1)

    # Empirical Distribution
    Q = repeat([1.0 / n], n)

    function objective(Ψ, Q, X, beta::Vector, dL)
        l = 0
        for i=1:n
            l = l .+ Q[i] * dL(Ψ[i], beta, X[i, :])
        end
        return l
    end

    function dobjective_dΨ(Ψ, Q, X, beta::Vector, dL)
        jacob = zeros(n, p)
        for i in 1:n
            jacob[i, :] = Q[i] * ∇dL(Ψ[i], beta, X[i, :])
        end
        return jacob
    end

    function dobjective_dQ(Ψ, Q, X, beta::Vector, dL)
        jacob = zeros(n, p)
        for i in 1:n
            jacob[i, :] = dL(Ψ[i], beta, X[i, :])
        end
        return jacob
    end

    function dobjective_dbeta(Ψ, Q, X, beta::Vector, dL)
        jacob = zeros(p, p)
        for i in 1:n
            jacob = jacob .+ Q[i] .* dL_dβ(Ψ[i], beta, X[i, :])
        end
        return jacob 
    end

    # Parameter mapping
    function B(Ψ, Q)
        #optim = optimize(beta -> sum_Lm(Ψ, Q, data, beta, Lm), repeat([0.0], p), LBFGS(), autodiff = :forward)
        #Optim.minimizer(optim)

        nlsolve(beta -> objective(Ψ, Q, X, beta, dL), repeat([0.0], p), autodiff = :forward, method = :newton).zero
    end

    function dB_dΨ(Ψ, Q, beta)
        -dobjective_dΨ(Ψ, Q, X, beta, dL) * inv(dobjective_dbeta(Ψ, Q, X, beta, dL))
    end

    function dB_dQ(Ψ, Q, beta)
        -dobjective_dQ(Ψ, Q, X, beta, dL) * inv(dobjective_dbeta(Ψ, Q, X, beta, dL))
    end

    function Δ(H, Y, Q̄)
        H .* (Y .- Q̄)
    end

    function normalizing_matrix(Ψ, Q, β, X)
        M = zeros(p, p)
        for i in 1:n
            M += Q[i] .* ddL(Ψ[i], β, X[i, :])
        end
        M⁻¹ = inv(M)
        return M⁻¹ 
    end

    function D(Ψ, Q̄, Q, g, β, Δ, X) 
        M⁻¹ = normalizing_matrix(Ψ, Q, β, X)

        eif = zeros(n, p)
        for i in 1:n
            eif[i, :] = -M⁻¹ * (dL(Ψ[i], β, X[i, :]) + ∇dL(Ψ[i], β, X[i, :]) * Δ[i])
        end

        return eif
    end

    function calculate_clever(H, H₀, H₁, Ψ, Q, β, X)
        clever = zeros(n, p)
        clever0 = zeros(n, p)
        clever1 = zeros(n, p)
        
        M⁻¹ = normalizing_matrix(Ψ, Q, β, X)

        for i in 1:n
            d = M⁻¹ * ∇dL(Ψ[i], β, X[i, :]) 
            clever[i, :] = H[i] .* d
            clever0[i, :] = H₀[i] .* d
            clever1[i, :] = H₁[i] .* d
        end

        return (clever, clever0, clever1)
    end
    
    function calculate_K(Ψ, Q, β, X)
        M⁻¹ = normalizing_matrix(Ψ, Q, β, X)
        K = zeros(n, p)
        for i = 1:n
            K[i, :] = M⁻¹ * dL(Ψ[i], β, X[i, :]) 
        end
        return K
    end

    # Combine all data
    data  = hcat(DataFrame(Y = Y, A = A), X)
    data0 = hcat(DataFrame(Y = Y, A = repeat([0.0], n)), X)
    data1 = hcat(DataFrame(Y = Y, A = repeat([1.0], n)), X)

    # Estimate g and get initial predictions
    if g == nothing
        g_model = glm(g_formula, data, Bernoulli(), LogitLink())
        g = predict(g_model)
    end

    # Estimate Q̄
    if Qbar === nothing
        if linear == true
            Q̄_model = lm(Qbar_formula, data)
        else 
            Q̄_model = glm(Qbar_formula, data, Bernoulli(), LogitLink())
        end

        Q̄  = predict(Q̄_model)
        Q̄₀ = predict(Q̄_model, data0)
        Q̄₁ = predict(Q̄_model, data1)
    else
        Q̄ = Qbar
        Q̄₀ = Qbar0
        Q̄₁ = Qbar1
    end

    Ψ = Q̄₁ - Q̄₀

    # Plug-in estimator
    β_plugin = B(Ψ, Q)

    # Clever covariates
    H₀ = -1 ./ (1. .- g)
    H₁ =  1 ./ (g)
    H  = ifelse.(A .== 1, H₁, H₀)

    function Q_fluctuation(ϵ, K, Q)
        Q_normalization = sum(exp.(K * ϵ) .* Q)
        return exp.(K * ϵ) .* Q ./ Q_normalization
    end

    function dQ_fluctuation_dϵ(ϵ, K, Q)
        Q_normalization = sum(exp.(K * ϵ) .* Q)
        return Q_fluctuation(ϵ, K, Q) .* (K .- Q .* K .* exp.(K * ϵ) ./ Q_normalization)
    end

    function model(ϵ, s, Q̄, Q̄₀, Q̄₁, clever, clever0, clever1, K, Q, Y; include_prior = false, linear = true)
        target = 0

        # Fluctuate Q̄
        if linear == true
            Q̄_fluctuated  = Q̄  .+ (s .^ 2) .* clever * ϵ
            Q̄₀_fluctuated = Q̄₀ .+ (s .^ 2) .* clever0 * ϵ
            Q̄₁_fluctuated = Q̄₁ .+ (s .^ 2) .* clever1 * ϵ 
        else
            Q̄_fluctuated  = logistic.(logit.(Q̄)  .+ clever * ϵ)
            Q̄₀_fluctuated = logistic.(logit.(Q̄₀) .+ clever0 * ϵ)
            Q̄₁_fluctuated = logistic.(logit.(Q̄₁) .+ clever1 * ϵ)
        end

        # Estimate Ψ
        Ψ_fluctuated = Q̄₁_fluctuated - Q̄₀_fluctuated

        Q_fluctuated = Q_fluctuation(ϵ, K, Q)

        # Estimate β
        β_fluctuated = B(Ψ_fluctuated, Q_fluctuated)
        
        if include_prior == true
            # Priors
            for i = 1:p
                target += loglikelihood(prior, β_fluctuated[i])
            end

            # Jacobian adjustment

            jacobian = dB_dΨ(Ψ_fluctuated, Q_fluctuated, β_fluctuated)' * (clever1 .- clever0) +
                dB_dQ(Ψ_fluctuated, Q_fluctuated, β_fluctuated)' * dQ_fluctuation_dϵ(ϵ, K, Q)

            target += log(abs(det(jacobian)))
        end

        # Likelihood
        for i in 1:n
            if linear == true
                target += loglikelihood(Normal(Q̄_fluctuated[i], s[i]), Y[i])
            else
                target += loglikelihood(Bernoulli(Q̄_fluctuated[i]), Y[i])
            end
        end

        target += sum(log.(Q_fluctuated))

        if isinf(target)
          β_fluctuated = repeat([0.0], length(ϵ))
          target = (target < 0) -1e8 : 1e8
        end

        return (β_fluctuated, target)
    end

    function mle(s, Q̄, Q̄₀, Q̄₁, clever, clever0, clever1, K, Q, Y; linear = true)
        Optim.minimizer(optimize(e -> -model(e, s, Q̄, Q̄₀, Q̄₁, clever, clever0, clever1, K, Q, Y; include_prior = false, linear = linear)[2], repeat([0.0], p)))
    end

    function yvariance(Q̄)
        data[:, :Y2] = (data[:, :Y] - Q̄) .^ 2
        y2_model = lm(var_formula, data)
        s = predict(y2_model)
        s = sqrt.(ifelse.(s .< 0, 1e-2, s))
        return s
    end

    # Calculate variance parameter
    s = repeat([1], n)
    if linear == true && bayes == true
        s = yvariance(Q̄)
    end

    Q̄_star  = Q̄ 
    Q̄₀_star = Q̄₀
    Q̄₁_star = Q̄₁
    Q_star = Q
    Ψ_star = Q̄₁_star - Q̄₀_star
    β_star = B(Ψ_star, Q_star)
    tmle_maxit = 100
    ϵ = repeat([0.0], p)
    for tmle_iter in 1:tmle_maxit
        # Update second clever covariate
        K = calculate_K(Ψ_star, Q_star, β_star, X)
        (clever, clever0, clever1) = calculate_clever(H, H₀, H₁, Ψ_star, Q_star, β_star, X)

        #ϵ = mle(repeat([1.0], n), Q̄_star, Q̄₀_star, Q̄₁_star, clever, clever0, clever1, K, Q_star, Y, linear = linear)
        ϵ = mle(s, Q̄_star, Q̄₀_star, Q̄₁_star, clever, clever0, clever1, K, Q_star, Y, linear = linear)
        
        if linear == true
            Q̄_star  = Q̄_star  .+ (s .^ 2) .* clever * ϵ
            Q̄₀_star = Q̄₀_star .+ (s .^ 2) .* clever0 * ϵ
            Q̄₁_star = Q̄₁_star .+ (s .^ 2) .* clever1 * ϵ
        else 
            Q̄_star  = logistic.(logit.(Q̄_star)  .+ clever  * ϵ)
            Q̄₀_star = logistic.(logit.(Q̄₀_star) .+ clever0 * ϵ)
            Q̄₁_star = logistic.(logit.(Q̄₁_star) .+ clever1 * ϵ)
        end

        #Q_star = Q_fluctuation(ϵ, K, Q_star)
        Q_star = Q
        Ψ_star = Q̄₁_star - Q̄₀_star 
        β_star = B(Ψ_star, Q_star)

        if maximum(abs.(ϵ)) < 1e-7
            break
        end
    end

    #return f(ϵ) = model(ϵ, s, Q̄_star, Q̄₀_star, Q̄₁_star, clever, clever0, clever1, K, Q_star, Y, include_prior = include_prior, linear = linear)
    #return f(ϵ) = Q_fluctuation(ϵ, K, Q_star)

    # Calculate EIF
    Δ_star = Δ(H, Y, Q̄_star)

    D_star = D(Ψ_star, Q̄_star, Q_star, g, β_star, Δ_star, X)

    # Calculate 95% confidence intervals
    β_star_se = mapslices(std, D_star, dims = 1)'
    β_star_lower = β_star .- quantile(Normal(), 0.975) .* β_star_se ./ sqrt(n)
    β_star_upper = β_star .+ quantile(Normal(), 0.975) .* β_star_se ./ sqrt(n)

    # Update clever covariates
    K = calculate_K(Ψ_star, Q_star, β_star, X)
    (clever, clever0, clever1) = calculate_clever(H, H₀, H₁, Ψ_star, Q_star, β_star, X)

    # Calculate variance parameter
    if linear == true && bayes == true
        s = yvariance(Q̄_star)
    end

    # Estimate marginal range of fluctuation model
    beta_mins = repeat([0.0], p)
    beta_maxes = repeat([0.0], p)
    for i in 1:p
      fmin = optimize(e -> model(e, s, Q̄_star, Q̄₀_star, Q̄₁_star, clever, clever0, clever1, K, Q_star, Y; include_prior = true, linear = linear)[1][i], repeat([0.0], p))
      fmax = optimize(e -> -1.0 * model(e, s, Q̄_star, Q̄₀_star, Q̄₁_star, clever, clever0, clever1, K, Q_star, Y; include_prior = true, linear = linear)[1][i], repeat([0.0], p))
      beta_mins[i]  = Optim.minimum(fmin)[1]
      beta_maxes[i]  = -1 * Optim.minimum(fmax)[1]
    end

    function metropolishastings(proposal_sd, iterations)
        samples = zeros(iterations, p)
        accepted = zeros(iterations)
        ll = zeros(iterations)
        β_post = zeros(iterations, p)

        samples[1, :] = repeat([0.0], p)
        (β, ll1) = model(samples[1, :], s, Q̄_star, Q̄₀_star, Q̄₁_star, clever, clever0, clever1, K, Q_star, Y, include_prior = include_prior, linear = linear)
        β_post[1, :] = β
        ll[1] = ll1
        accepted[1] = 0
        uniform = Uniform(0, 1)

        for i in 2:iterations
            # Proposal
            proposal = rand(rng, MvNormal(samples[i - 1, :], proposal_sd))

            (β, ll_proposal) = model(proposal, s, Q̄_star, Q̄₀_star, Q̄₁_star, clever, clever0, clever1, K, Q_star, Y, include_prior = include_prior, linear = linear)
            β_post[i, :] = β
            ll[i] = ll_proposal

            u = rand(rng, uniform)

            logratio = ll[i] - ll[i - 1]

            if logratio > log(u)
                samples[i, :] = proposal
                accepted[i] = 1
            else
                samples[i, :] = samples[i - 1, :]
                β_post[i, :] = β_post[i - 1, :]
                ll[i] = ll[i - 1]
                accepted[i] = 0
            end

            if i % 5_000 == 0
                println("MH Iteration ", i, "/", iterations)
            end
        end

        return (β_post, samples, mean(accepted))
    end

    β_post = nothing
    β_star_lower_bayes = nothing
    β_star_upper_bayes = nothing
    ϵ_post = nothing
    accepted = nothing
    β_star_bayes = nothing

    if bayes == true
        if proposal_sd === nothing
            # Estimate proposal sd
            println("Tuning Metropolis Hastings")
            current_accepted = 0.0
            max_tuning_iters = 20
            upper = 0.2
            lower = 0
            for iter in 1:max_tuning_iters
                proposal_sd = (upper + lower) / 2
                (_, _, new_accepted) = metropolishastings(proposal_sd, 1000)
                println("Proposal sd: ", proposal_sd, " acceptance: ", new_accepted)
                
                if new_accepted > 0.3 && new_accepted < 0.4
                    println("Accepting proposal sd ", proposal_sd, " with acceptance ", new_accepted)
                    break;
                elseif new_accepted >= 0.4
                    lower = proposal_sd
                else
                    upper = proposal_sd
                end

                current_accepted = new_accepted
            end
        end

        println("Running Metropolis Hastings")
        (β_post, ϵ_post, accepted) = metropolishastings(proposal_sd, iterations)

        β_star_bayes       = mapslices(x -> quantile(x, 0.5), β_post, dims = 1)'
        β_star_lower_bayes = mapslices(x -> quantile(x, 0.025), β_post, dims = 1)'
        β_star_upper_bayes = mapslices(x -> quantile(x, 0.975), β_post, dims = 1)'
    end

    # Check if any of the posterior draws are near the endpoints of the range
    # of the fluctuation model
    for i in 1:p
      print("beta[$(i)]: posterior range $(round(minimum(β_post[:, i]), sigdigits = 2)) - $(round(maximum(β_post[:, i]), sigdigits = 2)). Fluctuation model range: $(round(beta_mins[i], sigdigits = 2)) - $(round(beta_maxes[i], sigdigits = 2)).\n")
      if abs(maximum(β_post[:, i]) - beta_maxes[i]) < range_threshold || abs(minimum(β_post[:, i]) - beta_mins[i]) < range_threshold
        @warn "Posterior distribution for beta may be skewed due to choice of fluctuation model"
      end
    end

    return (
        epsilon = ϵ,
        beta_plugin = β_plugin,
        beta = β_star,
        beta_se = β_star_se,
        beta_lower = β_star_lower,
        beta_upper = β_star_upper,
        beta_post = β_post,
        beta_bayes = β_star_bayes,
        beta_lower_bayes = β_star_lower_bayes,
        beta_upper_bayes = β_star_upper_bayes,
        beta_mins = beta_mins,
        beta_maxes = beta_maxes,
        epsilon_post = ϵ_post,
        accepted = accepted,
        Psi = Ψ,
        Psi_star = Ψ_star,
        D_star = Δ_star
    )
end

end
