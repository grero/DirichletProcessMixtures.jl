function l-ratio(cids::Array{Int64,1}, X::Matrix{Float64}, Σ::Matrix{Float64},μ::Matrix{Float64})
    dims,nclusters = size(μ)
    clusterids = unique(cids)
    sort!(clusterids)
    dist = Mahalanobis(Σ)
    L = zeros(nclusters)
    for i in 1:size(X,2)
        d = mahalanobis(μ[:,c], X[:,i],Σ)
        L[cids[i]] += 1 - cdf(ChiSq(dims), d)
    end
    L
end

