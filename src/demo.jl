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

tic()
function iter_callback(mix::DirichletProcessMixtures.DPMM, iter::Int64, lower_bound::Float64)
    pl = sum(DirichletProcessMixtures.predictive_loglikelihood(mix, xtest)) /mix.M
    lb_log[iter] = lower_bound
    tl_log[iter] = pl
    toc()
    println("iteration $iter test likelihood=$pl, lower_bound=$lower_bound")
    tic()
end

niter = DirichletProcessMixtures.infer(model, x, maxiter, 1e-5; iter_callback=iter_callback)

using PyCall
@pyimport pylab

#convergence plot
#pylab.plot([1:niter], lb_log[1:niter]; color=[1., 0., 0.])
pylab.plot(1:niter, tl_log[1:niter]; color=(0., 0., 1.))

pylab.show()
    
z = DirichletProcessMixtures.map_assignments(model)
for k=1:T
xk = x[:, z .== k]
    pylab.scatter(xk[1, :], xk[2, :]; color=rand(3))
end
pylab.show()
