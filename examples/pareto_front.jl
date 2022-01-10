##
using LinearAlgebra
using Revise
include("../HopperHandshake.jl") # reloading HopperHandshake.jl will trigger lots of recompilation

## solve a random optimization to get all the necessary functions compiled
x0 = Designs.random_sample(1)
cost = Optimization.lsp_optimize(x0[:,1],.1)

## generate an initial guess for the linear scalarization problem corresponding to λ = .1
λ = .9
N = 20
x0 = Designs.random_sample(N^2)
cost = map(i->Optimization.cost(x0[:,i],λ),1:N^2)

# sort by cost
p = sortperm(cost)

# optimize a subset
cost = map(i->Optimization.lsp_optimize(x0[:,p[i]],λ;maxtime=10.)[1],1:N)

# select the best
x0 = x0[:,p[argmin(cost)]]

## next we attempt to solve the scalarization problem
minf, minx, ret = Optimization.lsp_optimize(x0,λ;maxtime=30.,ftol_rel = 1e-16)
xstar = minx
error,weight = Optimization.stationarity_test(minx;tol=1e-12)

## optimization code
f1(x) = Hopper.cost(x)                  # the value of f1(x) will be the optimization objective
df1(x) = Hopper.cost_grad(x)
f2(x) = Handshake.cost(x)               # the value of f2(x) will be constrained
df2(x) = Handshake.cost_grad(x)

N = 35                                  # number of iterations we will attempt
Δ = 0.1                                 # step change in f2 value
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
    if norm(minx-x[:,i-1]) < Δ^2
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

using Plots
cost_plot = scatter(hopper,handshake)

## select 3 different solutions to build
p = [Designs.unpack(x[:,i]) for i=1:size(x,2)]
idx = [6,22]

## what data do I want to make note of?

# Extension springs:
#   free length
#   spring rate
#   yield deflection
#   fixture angle

s1_len = [ExtensionSprings.free_length(p[i].s1) for i in idx]
s1_k = [ExtensionSprings.spring_rate(p[i].s1) for i in idx]
s1_yield = [ExtensionSprings.yield_deflection(p[i].s1,1.2) for i in idx]
s1_tension = [ExtensionSprings.mean_initial_tension(p[i].s1) for i in idx]


s2_len = [ExtensionSprings.free_length(p[i].s2) for i in idx]
s2_k = [ExtensionSprings.spring_rate(p[i].s2) for i in idx]
s2_yield = [ExtensionSprings.yield_deflection(p[i].s2,1.2) for i in idx]
s2_tension = [ExtensionSprings.mean_initial_tension(p[i].s2) for i in idx]


# Compression spring:
#   free length
#   spring rate
#   maximum deflection
#   yield deflection

s3_len = [p[i].s3.L0 for i in idx]
s3_k = [CompressionSprings.spring_rate(p[i].s3) for i in idx]
s3_max = [CompressionSprings.maximum_deflection(p[i].s3) for i in idx]
s3_yield = [CompressionSprings.yield_deflection(p[i].s3,1.2) for i in idx]

# Kinematics:
#   link lengths
s1_r = [p[i].s1_r for i in idx]
s2_r = [p[i].s2_r for i in idx]
l1 = [p[i].l1 for i in idx]
l2 = [p[i].l2 for i in idx]

## Save data in a Data Frame
using DataFrames
data = hcat(
    s1_len,
    s1_k,
    s1_yield,
    s1_tension,
    s2_len,
    s2_k,
    s2_yield,
    s2_tension,
    s3_len,
    s3_k,
    s3_yield,
    s3_max,
)
columns = [
    "s1 len",
    "s1 k",
    "s1 yield",
    "s1 tension",
    "s2 len",
    "s2 k",
    "s2 yield",
    "s2 tension",
    "s3 len",
    "s3 k",
    "s3 yield",
    "s3 max",
]

spring_data = DataFrame(data,columns)
CSV.write("nominal_spring_data.csv",spring_data)


data = hcat(s1_r, s2_r, l1, l2)
columns = ["s1 r", "s2 r", "l1", "l2"]
geometry_data = DataFrame(data,columns)
CSV.write("geometry_data.csv",geometry_data)
