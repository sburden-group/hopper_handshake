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
    grad += (1.0-λ)*Handshake.cost_grad(x)
end

""" Hessian of cost"""
function cost_hessian(x::Vector{T}, λ, h::T) where {T <: Real}
    hess = λ*Hopper.cost_hessian(x,h)
    hess += (1.0-λ)*Handshake.cost_hessian(x,h)
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
    active_set = [i for i=1:length(g) if g[i]>= -tol]
    g = g[active_set]
    ∇g = ∇g[active_set,:]

    model = JuMP.Model(SCS.Optimizer)
    set_optimizer_attribute(model, "verbose", 0)
    @variable(model, λ)
    @variable(model, ϵ)
    @variable(model, μ[i=1:length(g)])
    @constraint(model, ϵ >= 0)
    @constraint(model, λ >= 0)
    @constraint(model, λ <= 1)
    @constraint(model, μ .>= 0)
    y = @expression(model, λ*∇f1+(1-λ)*∇f2+∇g'*μ)
    @constraint(model, [ϵ;y] in SecondOrderCone())
    @objective(model, Min, ϵ)
    JuMP.optimize!(model)
    return (value(ϵ), value.(λ))
end

function second_order_test(x::Vector{T}; tol=0.) where {T<:Real}
    ϵ, λ = stationarity_test(x;tol=tol)
    ∇f1 = Hopper.cost_grad(x)
    Hf1 = Hopper.cost_hessian(x,1e-9)
    ∇f2 = Handshake.cost_grad(x)
    Hf2 = Handshake.cost_hessian(x,1e-9)
    g = Designs.constraints(x)
    ∇g = Designs.constraints_jacobian(x)
    Hg = FiniteDifferences.central_difference(Designs.constraints_jacobian,x,1e-9)
    active_set = [i for i=1:length(g) if g[i]>= -tol]
    g = g[active_set]
    ∇g = ∇g[active_set,:]

    model = JuMP.Model(SCS.Optimizer)
    set_optimizer_attribute(model, "verbose", 0)
    @variable(model, λ)
    @variable(model, ϵ)
    @variable(model, μ[i=1:length(g)])
    @constraint(model, ϵ >= 0)
    @constraint(model, λ >= 0)
    @constraint(model, λ <= 1)
    @constraint(model, μ .>= 0)
    y = @expression(model, λ*∇f1+(1-λ)*∇f2+∇g'*μ)
    @constraint(model, [ϵ;y] in SecondOrderCone())
    @objective(model, Min, ϵ)
    JuMP.optimize!(model)

    λ = value(λ)
    μ = value.(μ)
    first_order = λ*∇f1+(1-λ)*∇f2
    second_order = λ*Hf1 + (1-λ)*Hf2
    model = JuMP.Model(Ipopt.Optimizer)
    @variable(model, δx[i=1:length(x)])
    @variable(model, ϵ)
    @constraint(model, ϵ <= 0)
    @constraint(model, dot(δx,δx) == 1)
    @constraint(model, ∇g*δx .<= 0)
    @constraint(model, ϵ == dot(first_order,δx) + dot(δx,second_order*δx)/2)
    @objective(model, Min, ϵ)
    JuMP.optimize!(model)
    return value(ϵ), value.(δx)
end

function pareto_tangent(x::Vector{T}; tol=0.) where {T<:Real}
    ∇f1 = Hopper.cost_grad(x)
    Hf1 = Hopper.cost_hessian(x,1e-9)
    ∇f2 = Handshake.cost_grad(x)
    Hf2 = Handshake.cost_hessian(x,1e-9)
    g = Designs.constraints(x)
    ∇g = Designs.constraints_jacobian(x)
    Hg = FiniteDifferences.central_difference(Designs.constraints_jacobian,x,1e-9)
    active_set = [i for i=1:length(g) if g[i]>= -tol]

    model = JuMP.Model(SCS.Optimizer)
    set_optimizer_attribute(model, "verbose", 0)
    @variable(model, λ)
    @variable(model, ϵ)
    @variable(model, μ[i=1:length(active_set)])
    @variable(model, X[1:length(x),1:length(x)], PSD)
    @constraint(model, ϵ >= 0)
    @constraint(model, λ >= 0)
    @constraint(model, λ <= 1)
    @constraint(model, μ .>= 0)
    first_order = @expression(model, λ*∇f1+(1-λ)*∇f2+∇g[active_set,:]'*μ)
    @constraint(model, [ϵ;first_order] ∈ SecondOrderCone())
    @objective(model, Min, ϵ)
    JuMP.optimize!(model)

    λ = value(λ)
    μ = value.(μ)
    hess = λ*Hf1 + (1-λ)*Hf2 + sum([μ[i]*Hg[active_set[i],:,:] for i=1:length(active_set)])
    δx = hess\(∇f2-∇f1); δx = δx/norm(δx)
    model = JuMP.Model(Ipopt.Optimizer)
    @variable(model, h)
    @constraint(model, h>=0)
    forward = @expression(model, [g[i]+h*dot(∇g[i,:],δx)+h^2/2*dot(δx,Hg[i,:,:]*δx) for i=1:length(g)])
    backward = @expression(model, [g[i]-h*dot(∇g[i,:],δx)+h^2/2*dot(δx,Hg[i,:,:]*δx) for i=1:length(g)])
    @constraint(model, forward .<= 0)
    @constraint(model, backward .<= 0)
    @objective(model, Max, h)
    JuMP.optimize!(model)
    return value(h), δx
end

function optimize_control(x0;ftol_rel=1e-12,maxtime=10.,tol=0.)
    mechanical_params = x0[1:14]
    control_params = x0[15:end]
    f(x,grad) = begin
        try
            if length(grad) > 0
                grad[:] = cost_grad([mechanical_params...,x...],.5)[15:end]
            end
            return cost([mechanical_params...,x...],.5)
        catch
        end
    end
    g(result,x,grad) = begin
        try
            result[:] = Designs.nlconstraints([mechanical_params...,x...])[:]
            if length(grad) > 0
                grad[:,:] = Designs.nlconstraints_jacobian([mechanical_params...,x...])'[15:end,:]
            end
        catch e
            print(e)
            throw(e)
        end
    end
    opt = NLopt.Opt(:LD_SLSQP,length(control_params))
    opt.ftol_rel = ftol_rel
    lb,ub = Designs.bounds()
    opt.lower_bounds = lb[15:end]
    opt.upper_bounds = ub[15:end]
    opt.maxtime=maxtime
    opt.min_objective = f
    inequality_constraint!(opt, g, tol*ones(length(Designs.nlconstraints(x0))))
    (minf, minx, ret) = optimize(opt, control_params)
end

function lsp_optimize(x0,λ;ftol_rel=1e-12,maxtime=10.,tol=0.)
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
            result[:] = Designs.nlconstraints(x)[:]
            if length(grad) > 0
                grad[:,:] = Designs.nlconstraints_jacobian(x)'[:,:]
            end
        catch e
            print(e)
            throw(e)
        end
    end
    opt = NLopt.Opt(:LD_SLSQP,length(x0))
    opt.ftol_rel = ftol_rel
    lb,ub = Designs.bounds()
    opt.lower_bounds = lb
    opt.upper_bounds = ub
    opt.maxtime=maxtime
    opt.min_objective = f
    inequality_constraint!(opt, g, tol*ones(length(Designs.nlconstraints(x0))))
    (minf, minx, ret) = optimize(opt, x0)
end

function constraint_optimize(
            f1,             # function to optimize
            Df1,            # gradient of function to optimize
            f2,             # function to constraint
            Df2,            # jacobian of function to constrain
            x0::Vector,     # initial guess
            θ::Real;        # constraint bound
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
            result[:] = vcat(Designs.nlconstraints(x),f2(x)/f1(x)-tan(θ))
            if length(grad) > 0
                cgrad = (f1(x)*Df2(x)-f2(x)*Df1(x))/f1(x)^2
                grad[:,:] = vcat(Designs.nlconstraints_jacobian(x),cgrad')'[:,:]
            end
        catch e
            print(e)
            throw(e)
        end
    end
    opt = NLopt.Opt(:LD_SLSQP,length(x0))
    opt.ftol_rel = ftol_rel
    opt.maxtime=maxtime
    lb,ub = Designs.bounds()
    opt.lower_bounds = lb
    opt.upper_bounds = ub
    opt.min_objective = f
    inequality_constraint!(opt, g, zeros(length(Designs.nlconstraints(x0))+1))
    (minf, minx, ret) = optimize(opt, x0)
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
