"""
Compute mean and covariance of the clusters represented by `cids`.
"""
function get_cluster_parameters(cids::Array{Int64,1}, X::Matrix{Float64})
    dims,N = size(X)
    clusterids = unique(cids)
    sort!(clusterids)
    nclusters = length(clusterids)
    μ = Array{Vector{Float64},1}(nclusters)
    Σ = Array{Matrix{Float64},1}(nclusters)
    for (i,c) in enumerate(clusterids)
        _idx = find(cids .== c)
        _X = X[:,_idx]
        μ[i] = mean(_X,2)[:]
        cc = cov(_X')
        if isposdef(cc)
            Σ[i] = inv(cc)
        else
            Σ[i] = zeros(dims,dims)
        end
    end
    μ, Σ
end

function lratio(cids::Array{Int64,1}, X::Matrix{Float64})
    μ,Σ = get_cluster_parameters(cids,X)
    lratio(cids,X,Σ,μ)
end

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
    clusters = unique(cids)
    sort!(clusters)
    nclusters = length(μ)
    dims = size(X,1)
    dist = Dict(clusters[i]=>SqMahalanobis(Σ[i]) for i in 1:length(Σ))
    _μ = Dict(clusters[i] => μ[i] for i in 1:length(μ))
    L = Dict(c=>0.0 for c in clusters)
    chisq = Chisq(dims)
    for i in 1:size(X,2)
        @inbounds c = cids[i]
        for _c in clusters
            if _c != c
                d = Distances.evaluate(dist[_c], _μ[_c], view(X,:,i))
                @inbounds L[_c] += 1 - cdf(chisq, d)
            end
        end
    end
    L
end

function lratio(model::DPGMM, X::Matrix{Float64})
    cids = map_assignments(model)
    lratio(cids, X)
end
