module TargetedMSM

using LinearAlgebra
using DataFrames
using ForwardDiff
using GLM
using Statistics
using Turing
using Zygote
using ChainRulesCore
using NLsolve
using AdvancedMH

using Debugger

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(nlsolve), f, x0; kwargs...)
    result = nlsolve(f, x0; kwargs...)
    function nlsolve_pullback(Δresult)
        Δx = Δresult.zero
        x = result.zero
        _, f_pullback = rrule_via_ad(config, f, x)
        JT(v) = f_pullback(v)[2] # w.r.t. x
        # solve JT*Δfx = -Δx
        Δfx = nlsolve(v -> JT(v) + Δx, zero(x); kwargs...).zero
        ∂f = f_pullback(Δfx)[1] # w.r.t. f itself (implicitly closed-over variables)
        return (NoTangent(), ∂f, ZeroTangent())
    end
    return result, nlsolve_pullback
end

function multiple_treatments(Y, A, X, g_formula, Q̄_formula, p, L::Function, m::Function)

end

"""
    tmle(data, p::Integer, L, m, ℒ)

    m(beta::Vector, X::DataFrameRow)
    L(t, Z::DataFrameRow)
"""
function treatment_effect_modification(Y, A, X, g_formula, Q̄_formula, p, L::Function, m::Function)
    Lm(t, beta::Vector, X) = L(t, m(beta, X))

    # Derivatives of loss function
    dL(t, beta, X) = ForwardDiff.gradient(beta -> Lm(t, beta, X), beta)
    ddL(t, beta, X) = ForwardDiff.hessian(beta -> Lm(t, beta, X), beta)
    ∇dL(t, beta, X) = ForwardDiff.jacobian(t -> dL(first(t), beta, X), [t])

    # Number of observations
    n = size(Y, 1)

    # Empirical Distribution
    Q = repeat([1.0 / n], n)

    function sum_dL(Ψ, Q, X, beta::Vector, dL)
        l = 0
        for i=1:n
            l = l .+ Q[i] * dL(Ψ[i], beta, X[i, :])
        end
        return l
    end

    # Parameter mapping
    function B(Ψ, Q)
        #optim = optimize(beta -> sum_Lm(Ψ, Q, data, beta, Lm), repeat([0.0], p), LBFGS(), autodiff = :forward)
        #Optim.minimizer(optim)
        nlsolve(beta -> sum_dL(Ψ, Q, X, beta, dL), repeat([0.0], p), autodiff = :forward).zero
    end

    function Δ(H, Y, Q̄)
        H .* (Y .- Q̄)
    end

    function D(Ψ, Q̄, Q, g, β, Δ, X) 
        M = zeros(p, p)
        for i in 1:n
            M += Q[i] .* ddL(Ψ[i], β, X[i, :])
        end
        M⁻¹ = inv(M)

        eif = zeros(n, p)
        for i in 1:n
            eif[i, :] = -M⁻¹ * (dL(Ψ[i], β, X[i, :]) + ∇dL(Ψ[i], β, X[i, :]) * Δ[i])
        end

        return eif
    end

    # Combine all data
    data  = hcat(DataFrame(Y = Y, A = A), X)
    data0 = hcat(DataFrame(Y = Y, A = repeat([0.0], n)), X)
    data1 = hcat(DataFrame(Y = Y, A = repeat([1.0], n)), X)

    # Estimate g and get initial predictions
    g_model = glm(g_formula, data, Bernoulli(), LogitLink())
    g = predict(g_model)

    # Estimate Q̄
    Q̄_model = lm(Q̄_formula, data)

    Q̄ = predict(Q̄_model)
    Q̄₀ = predict(Q̄_model, data0)
    Q̄₁ = predict(Q̄_model, data1)

    Ψ = Q̄₁ - Q̄₀

    # Plug-in estimator
    β_plugin = B(Ψ, Q)

    # Clever covariates
    H₀ = -1 ./ (1. .- g)
    H₁ =  1 ./ (g)
    H  = ifelse.(A .== 1, H₁, H₀)

    clever = zeros(n, p)
    clever0 = zeros(n, p)
    clever1 = zeros(n, p)

    for i in 1:n
        d = ∇dL(Ψ[i], β_plugin, X[i, :])
        clever[i, :] = H[i] .* d
        clever0[i, :] = H₀[i] .* d
        clever1[i, :] = H₁[i] .* d
    end

    # Iterative algorithm
    update = fit(GeneralizedLinearModel, clever, Y, Normal(), IdentityLink(), offset = Q̄)
    ϵ = GLM.coef(update)

    Q̄_star  = Q̄  .+ clever * ϵ
    Q̄₀_star = Q̄₀ .+ clever0 * ϵ
    Q̄₁_star = Q̄₁ .+ clever1 * ϵ

    Ψ_star = Q̄₁_star - Q̄₀_star 

    β_star = B(Ψ_star, Q)

    Δ_star = Δ(H, Y, Q̄_star)

    D_star = D(Ψ_star, Q̄_star, Q, g, β_star, Δ_star, X)

    β_star_se = mapslices(std, D_star, dims = 1)'
    β_star_lower = β_star .- quantile(Normal(), 0.975) .* β_star_se ./ sqrt(n)
    β_star_upper = β_star .+ quantile(Normal(), 0.975) .* β_star_se ./ sqrt(n)

    # Bayesian TMLE
    @model function linear_fluctuation(p, Q̄, Q̄₀, Q̄₁, clever, clever0, clever1, K, Q, Y)
        s² ~ InverseGamma(2, 3)
        ϵ ~ filldist(Uniform(-1, 1), p)

        # Fluctuate Q̄
        Q̄_fluctuated = Q̄ .+ clever * ϵ
        Q̄₀_fluctuated = Q̄₀ .+ clever0 * ϵ
        Q̄₁_fluctuated = Q̄₁ .+ clever1 * ϵ
        
        ## Fluctuate Q
        Q_normalization = sum(exp.(K * ϵ) .* Q)
        Q_fluctuated  = exp.(K * ϵ) .* Q ./ Q_normalization
        
        ## Estimate Ψ
        Ψ_fluctuated = Q̄₁_fluctuated - Q̄₀_fluctuated

        # Estimate β
        β_fluctuated = B(Ψ_fluctuated, Q_fluctuated)
        #for i = 1:p
        #   Turing.@addlogprob! loglikelihood(Normal(0, 1), β_fluctuated[i])
        #end

        Y ~ MvNormal(Q̄_fluctuated, sqrt(s²))
        Turing.@addlogprob! sum(log.(Q_fluctuated))

        return β_fluctuated
    end

    # Compute second clever covariate
    K = zeros(n, p)
    for i = 1:n
        K[i, :] = dL(Ψ_star[i], β_star, X[i, :])
    end

    Turing.setadbackend(:zygote)
    model = linear_fluctuation(p, Q̄_star, Q̄₀_star, Q̄₁_star, clever, clever0, clever1, K, Q, Y)
    samples = sample(
        model,
        NUTS(250, 0.95),
        #MH(
        #    :s => x -> TruncatedNormal(x, 0.01, 0, Inf),
        #    :ϵ => AdvancedMH.RandomWalkProposal(MultivariateNormal(repeat([0.0], p), diagm(repeat([0.01, p]))))
        #),
        MCMCThreads(),
        250, 
        4,
        init_params = vcat([0.1], repeat([0.0], p))
    )

    beta_post = generated_quantities(model, samples)
    beta_post = mapreduce(permutedims, vcat, reshape(beta_post, prod(size(beta_post)), 1))
    beta_cis = mapslices(x -> quantile(x, [0.025, 0.975]), beta_post, dims = [1])

    return (
        epsilon = ϵ,
        beta = β_star,
        beta_se = β_star_se,
        beta_lower = β_star_lower,
        beta_upper = β_star_upper,
        beta_post = beta_post,
        beta_lower_bayes = beta_cis[1, :],
        beta_upper_bayes = beta_cis[2, :],
        samples = samples
    )
end

end
