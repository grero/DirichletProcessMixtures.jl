import Distributions.entropy

immutable TSBPMM
    α::Float64
    qv::Vector{Beta}
    π::Vector{Float64}
    z::Matrix{Float64}

    cluster_update::Function
    cluster_loglikelihood::Function
    object_loglikelihood::Function
    cluster_entropy::Function

    function TSBPMM(N::Int64, T::Int64, α::Float64,
            logpi::Vector{Float64}, z::Matrix{Float64},
            cluster_update::Function,
            cluster_loglikelihood::Function,
            object_loglikelihood::Function,
            cluster_entropy::Function)
        @assert length(logpi) == T "logpi must have size T"
        @assert size(z) == (N, T) "z has incostistent size"

        return new(α, [Beta(1., α) for i=1:T-1],
                logpi,
                z,
                cluster_update,
                cluster_loglikelihood,
                object_loglikelihood,
                cluster_entropy)
    end
end

function TSBPMM(N::Int64, T::Int64, α::Float64,
            cluster_update::Function,
            cluster_loglikelihood::Function,
            object_loglikelihood::Function,
            cluster_entropy::Function; random_init=false)
    logpi = zeros(T)
    z = zeros(N, T)

    if random_init
        rand!(logpi)
        ps = sum(logpi)
        logpi ./= ps
        logpi = log(logpi)

        rand!(z)
        for i=1:N
            zs = sum(z[i, :])
            z[i, :] ./= zs
        end
    else
        logpi[:] = -log(T)
        z[:, :] = 1. / T
    end

    return TSBPMM(N, T, α,
            logpi,
            z,
            cluster_update,
            cluster_loglikelihood,
            object_loglikelihood,
            cluster_entropy)
end

N(mix::TSBPMM) = size(mix.z, 1)
T(mix::TSBPMM) = size(mix.z, 2)

function infer(mix::TSBPMM, niter::Int64, ltol::Float64; iter_callback::Function = (oksa...) -> begin end)
    prev_lb = variational_lower_bound(mix)
    for iter=1:niter
        variational_update(mix)

        lb = variational_lower_bound(mix)

        iter_callback(mix, iter, lb)

        @assert lb >= prev_lb "Not monotone"
        if abs(lb - prev_lb) < ltol
            println("Converged")
            return iter
        end

        prev_lb = lb
    end

    return niter
end

function variational_update(mix::TSBPMM)
    z = zeros(T(mix))
    for i=1:N(mix)
        for k=1:T(mix)
            z[k] = mix.π[k] + mix.object_loglikelihood(k, i)
        end

        z .-= maximum(z)
        z = exp(z)
        z ./= sum(z)

        assert(abs(sum(z) - 1.) < 1e-7)

        mix.z[i, :] = z
    end

    ts = 0.
    for k=T(mix):-1:1
        zk = view(mix.z, :, k)
        mix.cluster_update(k, zk)
        zs = sum(zk)
        if k < T(mix)
            mix.qv[k] = Beta(1. + zs, mix.α + ts)
        end
        ts += zs
    end

    logpi!(mix.π, mix)
end

function variational_lower_bound(mix::TSBPMM)
    return loglikelihood(mix) + entropy(mix)
end

meanlog(beta::Beta) = digamma(beta.α) - digamma(beta.α + getfield(beta,2))
meanlogmirror(beta::Beta) = digamma(getfield(beta,2)) - digamma(beta.α + getfield(beta,2))
meanmirror(beta::Beta) = getfield(beta,2) / (beta.α + getfield(beta,2))
logmeanmirror(beta::Beta) = log(getfield(beta,2)) - log(beta.α + getfield(beta,2))

function logpi!(π::Vector{Float64}, mix::TSBPMM)
    r = 0.
    for k=1:T(mix)-1
        π[k] = meanlog(mix.qv[k]) + r
        r += meanlogmirror(mix.qv[k])
    end
    π[T(mix)] = r
end

function loglikelihood(mix::TSBPMM)
    ll = 0.

    ts = 0.
    for k=T(mix):-1:1
        zk = view(mix.z, :, k)
#        zk = mix.z[:, k]
        ll += mix.cluster_loglikelihood(k, zk)
        assert(!isnan(ll))

        zs = sum(zk)
        if k <= T(mix) - 1
            qv = mix.qv[k]
            ll += zs * meanlog(qv) + (mix.α+ts-1) * meanlogmirror(qv) - lbeta(1., mix.α)
            assert(!isnan(ll))
        end

        ts += zs
    end

    return ll
end

function entropy(mix::TSBPMM)
    ee = 0.
    ee += entropy(mix.z)

    for k=1:T(mix)
        if k < T(mix)
            ee += entropy(mix.qv[k])
        end
        ee += mix.cluster_entropy(k)
    end

    return ee
end

function map_assignments(mix::TSBPMM)
    z = zeros(Int64, N(mix))
    for i=1:N(mix)
        z[i] = indmax(view(mix.z, i, :))
    end
    return z
end

function pi!(π::Vector{Float64}, mix::TSBPMM)
    r = 0.
    for k=1:T(mix)-1
        qv = mix.qv[k]
        π[k] = exp(log(mean(qv)) + r)
        r += logmeanmirror(qv)
    end
    π[T(mix)] = exp(r)
    assert(abs(sum(π) - 1.) < 1e-7)
end

export TSBPMM, infer, variational_lower_bound, map_assignments, pi!, T
