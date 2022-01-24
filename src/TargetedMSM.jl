module TargetedMSM

using DataFrames
using ForwardDiff
using Optim
using GLM

using Debugger

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

    function sum_Lm(Ψ, Q, data::DataFrame, beta::Vector, Lm) 
        l = 0
        for i = 1:size(data, 1)
            l += Q[i] * Lm(Ψ[i], beta, data[i, :])
        end
        return l
    end

    # Parameter mapping
    function B(Ψ, Q)
        optim = optimize(beta -> sum_Lm(Ψ, Q, data, beta, Lm), repeat([0.0], p), LBFGS(), autodiff = :forward)
        Optim.minimizer(optim)
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

    # Number of observations
    n = size(Y, 1)

    # Empirical Distribution
    Q = repeat([1.0 / n], n)

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

    return (
        epsilon = ϵ,
        beta = β_star,
        beta_se = β_star_se,
        beta_lower = β_star_lower,
        beta_upper = β_star_upper
    )
end

end
