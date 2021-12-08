include("../HopperHandshake.jl")

# choose scalarization weight
λ = .75

# get a good initial guess using random sampling

x0 = Designs.random_sample(100) # 100 random Samples
cost = map(i->Optimization.cost(x0[:,i],λ),1:100)

# sort by cost
p = sortperm(cost)

# optimize the lowest ten for three seconds
cost = map(i->Optimization.lsp_optimize(x0[:,p[i]],λ;maxtime=3.)[1],1:10)

initial_guess = x0[:,p[argmin(cost)]]

minf,minx,ret = Optimization.lsp_optimize(initial_guess,λ;maxtime=60.)

# run stationarity test
Optimization.stationarity_test(minx)

# conclusion: minx is approximately locally pareto corresponding to λ ≈ 0.75