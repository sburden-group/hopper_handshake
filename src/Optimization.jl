module Optimization
using ..FiniteDifferences
using ..Designs
using ..Hopper
using ..Handshake
using LinearAlgebra
using ForwardDiff
using Distributions
using Random
using JuMP
using Ipopt
using NLopt
using SCS
using Plots


""" The scalarized objective function """
function cost(x::Vector{T}, λ) where {T <: Real}
    cost = λ*Hopper.cost(x)
    cost += (1.0-λ)*Handshake.cost(x)
    return cost
end

""" Gradient of cost """
function cost_grad(x::Vector{T}, λ) where {T <: Real}
    grad = λ*Hopper.cost_grad(x)
    grad += (1-λ)*Handshake.cost_grad(x)
end

""" Hessian of cost"""
function cost_hessian(x::Vector{T}, λ, h::T) where {T <: Real}
    hess = λ*Hopper.cost_hessian(x,h)
    hess += (1-λ)*Handshake.cost_hessian(x,h)
end


""" Jacobian of active affine constraints """
function active_constraints_jac(x::Vector{T}; tol=0.) where {T<:Real}
    r = Designs.constraints(x)
    active_set = [i for i=1:length(r) if -tol <= r[i]]
    return ForwardDiff.jacobian(Designs.constraints,x)[active_set,:]
end

""" Test for stationarity, formulated as a Convex program.
    Note that this requires affine constraints, which is only
    an approximation that is linear in x.
"""
function stationarity_test(x::Vector{T}; tol=0.) where {T<:Real}
    ∇f1 = Hopper.cost_grad(x)
    ∇f2 = Handshake.cost_grad(x)
    g = Designs.constraints(x)
    ∇g = Designs.constraints_jacobian(x)

    model = JuMP.Model(SCS.Optimizer)
    set_optimizer_attribute(model, "verbose", 1)
    @variable(model, λ)
    @variable(model, ϵ)
    @variable(model, μ[i=1:length(g)])
    @constraint(model, ϵ >= 0)
    @constraint(model, λ >= 0)
    @constraint(model, λ <= 1)
    @constraint(model, μ .>= 0)
    @constraint(model, μ.*g .== zeros(length(g)))
    y = @expression(model, λ*∇f1+(1-λ)*∇f2+∇g'*μ)
    @constraint(model, [ϵ;y] in SecondOrderCone())
    @objective(model, Min, ϵ)
    JuMP.optimize!(model)
    return (value(ϵ), value.(λ))
end

function lsp_optimize(x0,λ;ftol_rel=1e-12,maxtime=10.)
    f(x, grad) = begin
        try
            if length(grad) > 0
                grad[:] = cost_grad(x,λ)
            end
            return cost(x,λ)
        catch e
            print(e)
            throw(e)
        end
    end
    g(result,x,grad) = begin
        try
            result[:] = Designs.constraints(x)[:]
            if length(grad) > 0
                grad[:,:] = ForwardDiff.jacobian(
                    Designs.constraints, x    
                )'[:,:]
            end
        catch e
            print(e)
            throw(e)
        end
    end
    opt = NLopt.Opt(:LD_SLSQP,length(x0))
    opt.ftol_rel = ftol_rel
    opt.maxtime=maxtime
    opt.min_objective = f
    inequality_constraint!(opt, g, zeros(length(Designs.constraints(x0))))
    (minf, minx, ret) = optimize(opt, x0)
end

""" Generates a guesses for optimal x by randomly sampling n values,
    optimizing each for a short time (< seconds), and then returns
    the m<n best values."""
function initial_guess(λ::Float64,n::Int,m::Int)
    @assert m<n
    Random.seed!(42)
    x0 = Designs.random_sample(n^2)
    f0 = map(i->cost(x0[:,i],λ), 1:n^2)
    idx = sortperm(f0)
    x0 = x0[:,idx[1:n]]
    x = zeros((size(x0,1),n))
    f = zeros(n)
    for i=1:n
        try
            minf, minx, ret = lsp_optimize(x0[:,i],λ;ftol_rel=1e-12,maxtime=10.)
            x[:,i] = minx
            f[i] = minf
        catch e
            print(e)
            x[:,i] = x0[:,i]
            f[i] = cost(x[:,i],λ)
        end
    end
    idx = sortperm(f)
    return (x[:,idx[1:m]],f[idx[1:m]])
end

function constraint_optimize(
            f1,             # function to optimize
            Df1,            # gradient of function to optimize
            f2,             # function to constraint
            Df2,            # jacobian of function to constrain
            x0::Vector,     # initial guess
            ϵ::Real;        # constraint bound
            ftol_rel=1e-16, 
            maxtime=3.)
    f(x, grad) = begin
        try
            if length(grad) > 0
                grad[:] = Df1(x)
            end
            cost = f1(x)
            return cost
        catch e
            print(e)
            throw(e)
        end
    end
    g(result,x,grad) = begin
        try
            result[:] = vcat(Designs.constraints(x),f2(x)-ϵ)
            if length(grad) > 0
                grad[:,:] = vcat(Designs.constraints_jacobian(x),Df2(x)')'[:,:]
            end
        catch e
            print(e)
            throw(e)
        end
    end
    opt = NLopt.Opt(:LD_SLSQP,length(x0))
    opt.ftol_rel = ftol_rel
    opt.maxtime=maxtime
    opt.min_objective = f
    inequality_constraint!(opt, g, zeros(length(Designs.constraints(x0))+1))
    (minf, minx, ret) = optimize(opt, x0)
end

function constrain_handshake(initial, Δ, N)
    x0 = initial
    x = Array{Array{Float64,1},1}()
    θ = Array{Float64,1}()
    f1 = Array{Float64,1}()
    f2 = Array{Float64,1}()
    stationarity = Array{Float64,1}()
    append!(x,[x0])
    σ,λ = stationarity_test(x0)
    append!(θ,λ[1])
    append!(f1,Hopper.cost(x0))
    append!(f2,Handshake.cost(x0))
    append!(stationarity, σ)
    count = 0
    while count < N
        count += 1
        t0 = time()
        hopper_optimize(x,ϵ) = begin
            (minf,minx,ret) = constraint_optimize(
                Hopper.cost,
                Hopper.cost_grad,
                Handshake.cost,
                Handshake.cost_grad,
                x,
                ϵ;
                maxtime=60.,
            )
            print(ret); print("\n")
            return minx
        end
        ϵ = f2[end]-Δ
        minx = hopper_optimize(x[end],ϵ)
        (σ,λ) = stationarity_test(minx)
        print(string("Initial guess error: ", norm(x[end]-minx), "\n"))
        append!(x,[minx])
        append!(f1,Hopper.cost(minx))
        append!(f2,Handshake.cost(minx))
        print(string("x = ", x[end], "\n"))
        print(string("Hopper: ", f1[end], "\n"))
        print(string("Handshake: ", f2[end], "\n"))
        append!(θ,λ)
        append!(stationarity,σ)
        print(string("Stationarity test: ", stationarity[end], "\n"))
        print(string("Solution time: ", time()-t0, " seconds.\n"))
    end
    return f1, f2, reduce(hcat,x), stationarity
end


function constrain_hopper(initial, Δ, N)
    x0 = initial
    x = Array{Array{Float64,1},1}()
    θ = Array{Float64,1}()
    f1 = Array{Float64,1}()
    f2 = Array{Float64,1}()
    stationarity = Array{Float64,1}()
    append!(x,[x0])
    σ,λ = stationarity_test(x0)
    append!(θ,λ[1])
    append!(f1,Hopper.cost(x0))
    append!(f2,Handshake.cost(x0))
    append!(stationarity, σ)
    count = 0
    while count < N
        count += 1
        t0 = time()
        handshake_optimize(x,ϵ) = begin
            (minf,minx,ret) = constraint_optimize(
                Handshake.cost,
                Handshake.cost_grad,
                Hopper.cost,
                Hopper.cost_grad,
                x,
                ϵ;
                maxtime=60.,
            )
            print(ret); print("\n")
            return minx
        end
        ϵ = f1[end]-Δ
        minx = handshake_optimize(x[end],ϵ)
        (σ,λ) = stationarity_test(minx)
        print(string("Initial guess error: ", norm(x[end]-minx), "\n"))
        append!(x,[minx])
        append!(f1,Hopper.cost(minx))
        append!(f2,Handshake.cost(minx))
        print(string("x = ", x[end], "\n"))
        print(string("Hopper: ", f1[end], "\n"))
        print(string("Handshake: ", f2[end], "\n"))
        append!(θ,λ[1])
        append!(stationarity,σ)
        print(string("Stationarity test: ", stationarity[end], "\n"))
        print(string("Solution time: ", time()-t0, " seconds.\n"))
    end
    return f1, f2, reduce(hcat,x), stationarity
end

""" A pareto ranking algorithm, definitely naive. """
function pareto_ranking(f1,f2)
    points = [[f1[i],f2[i]] for i=1:length(f1)]
    index_set = [i for i=1:length(f1)]
    dominated = Array{Int}([])
    non_dominated = Array{Int}([])
    for j = 1:length(points)
        p = points[j]
        for i = 1:length(points)
            y = points[i]-p
            b = map(x->x<0,y)
            if b[1]&&b[2]
                append!(dominated,j)
                break
            end
            if i == length(points)
                append!(non_dominated,j)
            end
        end
    end
    return (dominated, non_dominated)
end

function max_potential_force(x)
    p = Designs.unpack(x)
    f(θ,grad) = begin
        q = vcat(0.,[θ[2]+.5*θ[1],θ[2]-.5*θ[1]])
        q[1] = -Hopper.constraints(q,p)[1]
        -max(abs.(Hopper.G\Hopper.potential_gradient(q,p))...)
    end
    opt = NLopt.Opt(:GN_DIRECT,2)
    opt.lower_bounds = [0.,-5*pi/180]
    opt.upper_bounds = [7pi/4,5*pi/180]
    opt.min_objective = f
    opt.maxtime=.25

    (minf,minx,ret) = optimize(opt, [pi/2,0.])
    return minf
end
end
