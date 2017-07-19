function lratio(cids::Array{Int64,1}, X::Matrix{Float64}, Σ::Matrix{Float64},μ::Matrix{Float64})
    dims,nclusters = size(μ)
    dist = Mahalanobis(Σ)
    L = zeros(nclusters)
    chisq = Chisq(dims)
    for i in 1:size(X,2)
        @inbounds c = cids[i]
        d = evaluate(dist, view(μ,:,c), view(X,:,i))
        @inbounds L[c] += 1 - cdf(chisq, d)
    end
    L
end

function lratio(cids::Array{Int64,1}, X::Matrix{Float64}, Σ::Vector{Matrix{Float64}},μ::Vector{Vector{Float64}})
    nclusters = length(μ)
    dims = size(X,1)
    dist = [Mahalanobis(Σ[c]) for c in 1:length(Σ)]
    L = zeros(nclusters)
    chisq = Chisq(dims)
    for i in 1:size(X,2)
        @inbounds c = cids[i]
        d = Distances.evaluate(dist[c], μ[c], view(X,:,i))
        @inbounds L[c] += 1 - cdf(chisq, d)
    end
    L
end

function lratio(model::DPGMM, X::Matrix{Float64})
    μ = Array{Array{Float64,1}}(length(model.theta))
    Σ = Array{Matrix{Float64}}(length(model.theta))
    #draw from the prior
    for c in 1:length(μ)
       μ[c], Σ[c] = rand(model.theta[c])
    end
    cids = map_assignments(model)
    lratio(cids, X, Σ, μ)
end
