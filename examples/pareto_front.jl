##
using LinearAlgebra
using Revise
using Plots
include("../HopperHandshake.jl") # reloading HopperHandshake.jl will trigger lots of recompilation

## brute force the problem
N = 10000
x = Designs.random_sample(N)
hopper = map(i->Hopper.cost(x[:,i]),1:N)
handshake = map(i->Handshake.cost(x[:,i]),1:N)
dominated, non_dominated = Optimization.pareto_ranking(hopper,handshake)

scatter(hopper[non_dominated], handshake[non_dominated])
scatter(x[1,non_dominated],x[2,non_dominated])

using DataFrames
using CSV
data = hcat(hopper,handshake,x[:,non_dominated]')
df = DataFrame(data,columns=["Hopper","Handshake","L1","L2"])
CSV.write(df,"no_spring_pareto.csv")