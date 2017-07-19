using Base.Test
using StatsBase
using DirichletProcessMixtures
using Distributions
using ConjugatePriors

import ConjugatePriors.NormalWishart

function ball(N::Int64, x::Float64, y::Float64)
    return randn(2, N) .+ [x, y]
end

function balley(M::Int64, R::Float64)
    return hcat(ball(M, 0., 0.),
            ball(M,  R,  R),
            ball(M,  R, -R),
            ball(M, -R,  R),
            ball(M, -R, -R))
end
srand(1234)
B = 60
x = balley(B, 3.)
xtest = balley(B, 3.)

N = B * 5
M = N

prior = NormalWishart(zeros(2), 1e-7, eye(2) / 4.0, 4.0001)

T = 20
maxiter = 4000
model = DirichletProcessMixtures.DPGMM(size(x,2), T, 1e-1, prior)

lb_log = zeros(maxiter)
tl_log = zeros(maxiter)

niter = DirichletProcessMixtures.infer(model, x, maxiter, 1e-5)

    
z = DirichletProcessMixtures.map_assignments(model)
C = countmap(z)
@test C[1] == 123
@test C[3] == 57
@test C[4] == 59
@test C[6] == 61
@test niter == 117

ll = DirichletProcessMixtures.lratio(model, x)
@test ll[1] ≈ 70.09237663387312
@test ll[3] ≈ 32.30968894374262
@test ll[4] ≈ 34.60151572248789
@test ll[6] ≈ 35.17672736953846
