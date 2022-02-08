##
using DataFrames
using CSV
using LinearAlgebra
using Revise
using Random
include("../HopperHandshake.jl") # reloading HopperHandshake.jl will trigger lots of recompilation

## generate random samples
N = 500
Random.seed!(42)
random_samples = Designs.random_sample(N)
random_hopper = map(i->Hopper.cost(random_samples[:,i]),1:N)
random_handshake = map(i->Handshake.cost(random_samples[:,i]),1:N)

## generate initial optimization solutions
λ = 0.9
cost = map(i->λ*random_hopper[i]+(1-λ)*random_handshake[i],1:length(random_hopper))

# sort by cost
p = sortperm(cost)

# select the best
x0 = random_samples[:,p[argmin(cost)]]

# dummy optimization to trigger compilation
minf, minx, ret = Optimization.lsp_optimize(x0,λ;maxtime=1.,ftol_rel = 1e-16)

# next we attempt to solve the scalarization problem
minf, minx, ret = Optimization.lsp_optimize(x0,λ;maxtime=30.,ftol_rel = 1e-16)
xstar1 = minx
θ1 = atan(Handshake.cost(xstar1)/Hopper.cost(xstar1))
error,weight = Optimization.stationarity_test(minx;tol=1e-12)

λ = 0.1
cost = map(i->λ*random_hopper[i]+(1-λ)*random_handshake[i],1:length(random_hopper))

# sort by cost
p = sortperm(cost)

# select the best
x0 = random_samples[:,p[argmin(cost)]]

# next we attempt to solve the scalarization problem
minf, minx, ret = Optimization.lsp_optimize(x0,λ;maxtime=30.,ftol_rel = 1e-16)
xstar2 = minx
θ2 = atan(Handshake.cost(xstar2)/Hopper.cost(xstar2))
error,weight = Optimization.stationarity_test(minx;tol=1e-12)

## optimization code
f1(x) = Hopper.cost(x)                  # the value of f1(x) will be the optimization objective
df1(x) = Hopper.cost_grad(x)
f2(x) = Handshake.cost(x)               # the value of f2(x) will be constrained
df2(x) = Handshake.cost_grad(x)

N = 15                                  # number of iterations we will attempt
sig = [1/(1+exp(-4x)) for x in range(-1,1,length=N)]
θ = θ1 .+ (θ2-θ1)*sig
x = zeros((length(minx),N+2))             
x[:,1] = xstar1

for i=1:N
    minf,minx,ret = Optimization.constraint_optimize(
                        f1,
                        df1,
                        f2,
                        df2,
                        x[:,i],
                        θ[i];
                        ftol_rel=1e-16,
                        maxtime=60.
                    )
    x[:,i+1] = minx
end
x[:,end] = xstar2

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

# plot the random samples
idx = filter(i->random_hopper[i]<4 && random_handshake[i]<4, 1:length(random_hopper))
scatter!(cost_plot,random_hopper[idx],random_handshake[idx];label="random samples",markershape=:cross)

xlabel!(cost_plot,"Hopper cost")
ylabel!(cost_plot,"Handshake cost")

## Save all the data from this figure!
columns = vcat("hopper","shaker",["x$(i)" for i=1:17]...)
df = DataFrame(hcat(hopper,handshake,x'),columns)
CSV.write("pareto_front.csv", df)

# random samples
df = DataFrame(hcat(random_hopper,random_handshake,random_samples'),columns)
CSV.write("random_samples.csv", df)

# no spring designs
costs = vcat(minitaur_cost', [Hopper.cost(nospring_optimized), Handshake.cost(nospring_optimized)]')

df = DataFrame(hcat(costs,vcat(minitaur',nospring_optimized')), columns)
CSV.write("nosprings.csv", df)

## select 3 different solutions to build
p = [Designs.unpack(x[:,i]) for i=1:size(x,2)]
idx = [1,7,30]
scatter!(cost_plot,hopper[idx],handshake[idx];label="efficient samples",markershape=:star,markersize=7)
savefig(cost_plot,"cost_plot")

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
data = hcat(
    idx,
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
    "index",
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
    "s3 max"
]

spring_data = DataFrame(data,columns)
# CSV.write("nominal_spring_data.csv",spring_data)


data = hcat(idx, s1_r, s2_r, l1, l2)
columns = ["idx","s1 r", "s2 r", "l1", "l2"]
geometry_data = DataFrame(data,columns)
# CSV.write("geometry_data.csv",geometry_data)
