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
λ1 = 0.95
cost = map(i->λ1*random_hopper[i]+(1-λ1)*random_handshake[i],1:length(random_hopper))
# sort by cost
p = sortperm(cost)
# select the best
x0 = random_samples[:,p[argmin(cost)]]
minf, minx, ret = Optimization.lsp_optimize(x0,λ1;maxtime=30.,ftol_rel = 1e-16)
xstar1 = minx
error,weight = Optimization.stationarity_test(minx;tol=1e-12)
θ1 = atan(Handshake.cost(xstar1)/Hopper.cost(xstar1))

λ2 = 0.05
cost = map(i->λ2*random_hopper[i]+(1-λ2)*random_handshake[i],1:length(random_hopper))
# sort by cost
p = sortperm(cost)
# select the best
x0 = random_samples[:,p[argmin(cost)]]
minf, minx, ret = Optimization.lsp_optimize(x0,λ2;maxtime=30.,ftol_rel = 1e-16)
xstar2 = minx
error,weight = Optimization.stationarity_test(minx;tol=1e-12)
θ2 = atan(Handshake.cost(xstar2)/Hopper.cost(xstar2))

## optimization code
f1(x) = Hopper.cost(x)
df1(x) = Hopper.cost_grad(x)
f2(x) = Handshake.cost(x)
df2(x) = Handshake.cost_grad(x)
N = 20
θ = range(θ1,θ2,N)
x = zeros((length(minx),N))
x[:,1] = xstar1
for i=2:N
    minf,minx,ret = Optimization.constraint_optimize(
        f1,df1,f2,df2,x[:,i-1],θ[i];ftol_rel=1e-16,maxtime=30.
    )
    x[:,i] = minx
end

##

# evaluate costs
hopper = map(i->Hopper.cost(x[:,i]),1:size(x,2))
handshake = map(i->Handshake.cost(x[:,i]),1:size(x,2))

# filter data by error and sort by hopper
stationarity = map(i->Optimization.stationarity_test(x[:,i];tol=1e-12),1:size(x,2))
error = map(i->stationarity[i][1],1:length(stationarity))
weight = map(i->stationarity[i][2],1:length(stationarity))

idx = filter(i->abs(error[i])<1e-3,1:N)
p = sortperm(hopper[idx])

# create a figure of the objective space
using Plots

# plot the pareto front
cost_plot = scatter(hopper[idx[p]],handshake[idx[p]];label="pareto points")

# save the data
columns = vcat("hopper","shaker",["x$(i)" for i=1:5]...)
df = DataFrame(hcat(hopper,handshake,x'),columns)
CSV.write("nospring_front.csv", df)