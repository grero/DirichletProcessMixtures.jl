import Distributions.MvNormalStats, Distributions.suffstats, Distributions.mean
import ConjugatePriors.NormalWishart

type DPGMM <: DPMM
    α::Float64
    qv::Vector{Beta{Float64}}
    π::Vector{Float64}
    z::Matrix{Float64}
    prior::NormalWishart{Float64}
    theta::Vector{NormalWishart{Float64}}
    M::Int64
    N::Int64
end

function DPGMM(N::Int64, M::Int64, α::Float64, logpi::Vector{Float64}, z::Matrix{Float64},prior::Distribution)
    length(logpi) == M || throw(ArgumentError("`logpi` must have size `M`"))
    size(z) == (N,M) || throw(ArgumentError("`z` has inconsistent size"))
    return DPGMM(α, [Beta(1.0, α) for i in 1:M-1], logpi, z,prior,[prior for k in 1:M],M,N)
end

function DPGMM(N::Int64, M::Int64, α::Float64,prior::Distribution;random_init=false)
    logpi = zeros(M)
    z = zeros(N, M)
    if random_init
        rand!(logpi)
        ps = sum(logpi)
        logpi ./= ps
        logpi = log.(logpi)

        rand!(z)
        for i=1:N
            zs = sum(z[i, :])
            z[i, :] ./= zs
        end
    else
        logpi .= -log.(M)
        z .= 1. / M
    end
    return DPGMM(N, M, α, logpi, z, prior)
end

function suffstats(D::Type{MvNormal}, x::Matrix{Float64}, w::AbstractArray{Float64})
    d = size(x, 1)
    n = size(x, 2)

    tw = sum(w)
    s = zeros(d)
    for i=1:n
        for j in 1:d
            s[j] += x[j, i] .* w[i]
        end
    end
    m = s * inv(tw)
    z = (x.-m).* sqrt.(w)' # subtract dim 1, multiply dim 2
    s2 = A_mul_Bt(z, z)

    MvNormalStats(s, m, s2, tw)
end

function posterior_cool(prior::NormalWishart, ss::MvNormalStats)
    if ss.tw < eps() return prior end

    mu0 = prior.mu
    kappa0 = prior.kappa
    TC0 = prior.Tchol
    nu0 = prior.nu

    kappa = kappa0 + ss.tw
    nu = nu0 + ss.tw
    mu = (kappa0.*mu0 + ss.s) ./ kappa

    Lam0 = inv(TC0)
    z = prior.zeromean ? ss.m : ss.m - mu0
    Lam = inv(Lam0 + ss.s2 + kappa0*ss.tw/kappa*(z*z'))
    return NormalWishart(mu, kappa, cholfact(Hermitian(Lam)), nu)
end

function expected_logdet(nw::NormalWishart)
    logd = 0.0

    for i=1:nw.dim
        _q = digamma(0.5 * (nw.nu + 1 - i))
        logd += _q
    end

    logd += nw.dim * log(2)
    logd += logdet(nw.Tchol)::Float64

    return logd
end

function expected_T(nw::NormalWishart)
    return nw.Tchol[:U]' * nw.Tchol[:U] * nw.nu
end

function mean(nw::NormalWishart)
    T = expected_T(nw)
    return MvNormalCanon(T * nw.mu, T)
end

function lognorm(nw::NormalWishart)
    return (nw.dim / 2) * (log(2 * pi) - log(nw.kappa)) + (nw.nu / 2) * logdet(nw.Tchol) + (nw.dim * nw.nu / 2) * log(2.) + lpgamma(nw.dim, nw.nu / 2)
end

function entropy(nw::NormalWishart)
    en = 0.

    logd = expected_logdet(nw)

    en -= logd
    en -= nw.dim * (log(nw.kappa) - log(2*pi))
    en += nw.dim
    en /= 2

    en -= 0.5 * (nw.nu - nw.dim - 1) * logd
    en += nw.nu * nw.dim / 2
    en += 0.5 * nw.nu * logdet(nw.Tchol) + 0.5 * nw.nu * nw.dim * log(2.) + lpgamma(nw.dim, nw.nu/2)

    return en
end

function marginal_loglikelihood(prior::NormalWishart, posterior::NormalWishart, n::Float64)
    d = prior.dim
    return lognorm(posterior) - lognorm(prior) - (n*d/2) * log(2 * pi)
end

function cluster_update(model::DPGMM, k::Int64, z::AbstractArray{Float64},x::Matrix{Float64})
    nw = posterior_cool(model.prior, suffstats(MvNormal, x, z))
    model.theta[k] = nw
end

function cluster_loglikelihood(model::DPGMM, k::Int64, z::AbstractArray{Float64},x::Matrix{Float64})
    post = posterior_cool(model.prior, suffstats(MvNormal, x, z))
    n = sum(z)

    ll = -lognorm(model.prior)

    nw = model.theta[k]
    ll -= nw.dim * n / 2 * log(2*pi)

    ll += (post.nu - post.dim)/2 * expected_logdet(nw)
    W = nw.Tchol[:U]' * nw.Tchol[:U]
    ll -= nw.nu * trace(inv(post.Tchol) * W) / 2

    dm = nw.Tchol[:U] * (nw.mu - post.mu)
    ll -= post.kappa * (nw.dim / nw.kappa + nw.nu * dot(dm, dm)) / 2
    return ll
end

function object_loglikelihood(model::DPGMM, k::Int64, i::Int64,x::Matrix{Float64})
    nw = model.theta[k]
    ll = -nw.dim/2 * log(2*pi) + expected_logdet(nw)/2
    dx = nw.Tchol[:U] * (x[:, i] - nw.mu)
    ll -= (nw.dim / nw.kappa + nw.nu * dot(dx, dx)) / 2

    return ll
end

cluster_entropy(model::DPGMM, k::Int64) = entropy(model.theta[k])

function predictive_loglikelihood(model::DPGMM, xt::Matrix{Float64})
    nt = size(xt, 2)
    logp = zeros(nt)

    π = zeros(model.M)
    pi!(π, model)

    pred = map(predictive, model.theta)

    for i=1:nt
        for k=1:model.M
            logp[i] += π[k] * pdf(pred[k], xt[:, i])
        end
    end
    logp = log.(logp)

    return logp
end

function gaussian_mixture(prior::NormalWishart, T::Int64, alpha::Float64, x::Matrix{Float64})
    dim, N = size(x)
    theta = Array{NormalWishart}(T)

    for k=1:T
        theta[k] = prior
    end


    mm = TSBPMM(N, T, alpha, cluster_update,
            cluster_loglikelihood,
            object_loglikelihood,
            cluster_entropy; random_init=true)


    return mm, theta, predictive_loglikelihood
end

export gaussian_mixture
