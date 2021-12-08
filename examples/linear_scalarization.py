"""
To use this, you MUST have Julia version >= 1.7.0 installed, as well as PyJulia.
Follow the instructions at https://pyjulia.readthedocs.io/en/stable/
"""
#%% essential imports
import julia
import numpy as np

#%% configure julia environment
JULIA_RUNTIME = "C:\\Users\\spear\\AppData\\local\\programs\\julia-1.7.0\\bin\\julia.exe"
jl = julia.core.Julia(runtime=JULIA_RUNTIME)

# now we should be able to import "Main", which is a python wrapper for the Julia REPL
from julia import Pkg, Main
PROJECT_RELATIVE_PATH = "../"

# next we need to activate this project in julia, and include the source code
Pkg.activate(PROJECT_RELATIVE_PATH)
Main.include(PROJECT_RELATIVE_PATH+"HopperHandshake.jl")

# %% the environment is set up, and we can begin working with the julia modules

# choose scalarization weight
weight = .75

# get a good initial guess using random sampling
x0 = Main.Designs.random_sample(100)
cost = [Main.Optimization.cost(x0[:,i],weight) for i in range(100)]

# sort indices of x0[:,i] by cost
p = sorted(range(100), key = lambda i: cost[i])

# optimize the lowest ten for 5 seconds each
cost = [Main.Optimization.lsp_optimize(x0[:,p[i]],weight,maxtime=5.)[0] for i in range(10)]

# %% use the best result as our initial guess in the optimization
initial_guess = x0[:,p[np.argmin(cost)]]

minf, minx, ret = Main.Optimization.lsp_optimize(initial_guess,weight,maxtime=60.)

# run stationarity test
error, weight = Main.Optimization.stationarity_test(minx)

# conclusion: minx is approximately locally pareto corresponding to weight 0.75
