##
using DataFrames
using CSV
using LinearAlgebra
using Revise
using Random
include("../HopperHandshake.jl") # reloading HopperHandshake.jl will trigger lots of recompilation

## generate random samples
N = 1000
Random.seed!(42)
random_samples = Designs.random_sample(N)
random_hopper = map(i->Hopper.cost(random_samples[:,i]),1:N)
random_handshake = map(i->Handshake.cost(random_samples[:,i]),1:N)

## generate initial guess for optimization
λ = 0.05
cost = map(i->λ*random_hopper[i]+(1-λ)*random_handshake[i],1:length(random_hopper))

# sort by cost
p = sortperm(cost)

# select the best
x0 = random_samples[:,p[argmin(cost)]]

# dummy optimization to trigger compilation
minf, minx, ret = Optimization.lsp_optimize(x0,λ;maxtime=2.,ftol_rel = 1e-16)

## next we attempt to solve the scalarization problem
minf, minx, ret = Optimization.lsp_optimize(x0,λ;maxtime=30.,ftol_rel = 1e-16)
xstar = minx
error,weight = Optimization.stationarity_test(minx;tol=1e-12)

## optimization code
f2(x) = Hopper.cost(x)                  # the value of f1(x) will be the optimization objective
df2(x) = Hopper.cost_grad(x)
f1(x) = Handshake.cost(x)               # the value of f2(x) will be constrained
df1(x) = Handshake.cost_grad(x)

N = 60                                  # number of iterations we will attempt
Δ = 0.025                                 # step change in f2 value
x = zeros((length(minx),N))             
x[:,1] = xstar                          

for i=2:N
    ϵ = f2(x[:,i-1])-Δ
    minf,minx,ret = Optimization.constraint_optimize(
                        f1,
                        df1,
                        f2,
                        df2,
                        x[:,i-1],
                        ϵ;
                        ftol_rel=1e-16,
                        maxtime=60.
                    )
    if f1(minx)-f1(x[:,i-1]) > 2Δ
        Δ = Δ/2
        minf,minx,ret = Optimization.constraint_optimize(
                            f1,
                            df1,
                            f2,
                            df2,
                            x[:,i-1],
                            ϵ;
                            ftol_rel=1e-16,
                            maxtime=60.
                        )
    end
    if norm(minx-x[:,i-1]) < 1e-2
        x = x[:,1:i-1]
        break
    end
    x[:,i] = minx
end

##

# evaluate costs
hopper = map(i->Hopper.cost(x[:,i]),1:size(x,2))
handshake = map(i->Handshake.cost(x[:,i]),1:size(x,2))

# sort by hopper
p = sortperm(hopper)
hopper[:] = hopper[p]
handshake[:] = handshake[p]
x[:,:] = x[:,p]

stationarity = map(i->Optimization.stationarity_test(x[:,i];tol=1e-9),1:size(x,2))
error = map(i->stationarity[i][1],1:length(stationarity))
weight = map(i->stationarity[i][2],1:length(stationarity))

# create a figure of the objective space
using Plots

# plot the pareto front
cost_plot = scatter(hopper,handshake;label="pareto points")

# save the data
