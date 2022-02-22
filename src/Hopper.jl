module Hopper

using ..Designs
using ..FiniteDifferences
using NLopt
using NLsolve
using Plots
using ForwardDiff: jacobian, gradient, JacobianConfig
using ForwardDiff: gradient, GradientConfig
using ForwardDiff: hessian, HessianConfig, Chunk
using LinearAlgebra
using Random: rand


"""
BEGIN MODEL DEFINITION, DYNAMICS, CONTROL
"""
const m = 2.0
const Jm = .5/2*(.087^2+.08^2) # rough approximation motor inertia from thick-walled cylinder model.

# constants related to (unconstrained) statespace
const body_idx = 1
const θ1_idx = 2
const θ2_idx = 3
const n = 3 # number of unconstrained configuration variables

# link inertias: for now, neglect entirely

# motor constants (am I including motor dynamics in the model?)
const R = 0.186
const Kv = (2pi*100.0/60) # (Rad/s) / Volt
const Ke = 1/Kv           # Volt / (rad/s)

# kinematics
function hip_foot_angle(q::Vector{T}) where {T <: Real}
    (q[θ1_idx]+q[θ2_idx])/2.0
end

function interior_leg_angle(q::Vector{T}) where {T <: Real}
    (q[θ1_idx]-q[θ2_idx])/2.0
end

function leg_length(q::Vector{T}, p::Designs.Params) where {T <: Real}
    ϕ = interior_leg_angle(q)
    p.l1*cos(ϕ)+sqrt(p.l2^2-(p.l1*sin(ϕ))^2)+.03
end

function potential_energy(q::Vector{T},p::Designs.Params) where {T <: Real}
    g = 9.81
    l = leg_length(q,p)
    θ1 = q[θ1_idx]
    θ2 = q[θ2_idx]

    s1_fl = Designs.ExtensionSprings.free_length(p.s1)
    joint_location = Array{T}([p.l1*sin(θ1),-p.l1*cos(θ1)])
    r = (p.l1+s1_fl+.015)
    fixed_end = Array{T}([r*sin(p.s1_r),-r*cos(p.s1_r)]) 
    free_end = joint_location + .015*(fixed_end-joint_location)/norm(fixed_end-joint_location)
    δx = fixed_end-free_end
    s1_energy = Designs.ExtensionSprings.spring_energy(p.s1,norm(δx)-s1_fl)

    s2_fl = Designs.ExtensionSprings.free_length(p.s2)
    joint_location = Array{T}([p.l1*sin(θ2),-p.l1*cos(θ2)])
    r = (p.l1+s2_fl+.015)
    fixed_end = Array{T}([r*sin(p.s2_r),-r*cos(p.s2_r)]) 
    free_end = joint_location + .015*(fixed_end-joint_location)/norm(fixed_end-joint_location)
    δx = fixed_end-free_end
    s2_energy = Designs.ExtensionSprings.spring_energy(p.s2,norm(δx)-s2_fl)

    s3_energy = Designs.CompressionSprings.spring_energy(p.s3,p.l1+p.l2-l)

    return T(m*g*q[body_idx]+s1_energy+s2_energy+s3_energy)
end

function kinetic_energy(q::Vector,qdot::Vector,p::Designs.Params)
    return 0.5*m*qdot[body_idx]^2+0.5*Jm*(qdot[θ1_idx]^2+qdot[θ2_idx]^2)
end

# mass matrix
const M = diagm([m,Jm,Jm])
const Minv = inv(M)

# ∇V for the model ( can simplify greatly since it should be representable as -Kx-F for fixed K,F in this model )
function potential_gradient(q::Vector{T},p::Designs.Params) where T
    f(x) = potential_energy(x,p)
    cfg = GradientConfig(f, q, Chunk{n}())
    return gradient(f,q,cfg)
end

# kinematic constraint on configuration due to foot contact at (0,0)
function constraints(q::Vector{T},p::Designs.Params) where T<:Real
    θ = hip_foot_angle(q)
    l = leg_length(q,p)
    return [q[body_idx]-l*cos(θ), l*sin(θ)]
end

# Constraint forces
function A_jacobian(q::Vector{T},p::Designs.Params) where T<:Real
    f(q) = constraints(q,p)
    cfg = JacobianConfig(f, q, Chunk{n}())
    jacobian(f,q,cfg)
end

"""
Calculates the derivative of DA(q)
"""
function A_jacobian_prime(q::Vector{T},qdot::Vector{T},p::Designs.Params) where T<:Real
    rows = 2 # number of constraint equations
    f(x) = A_jacobian(x,p)
    cfg = JacobianConfig(f, q, Chunk{n}())
    reshape(jacobian(f,q,cfg)*qdot,rows,n)
end

# input force map ( is actually constant in this model )
const G = [
    0. 0.
    1. 0.
    0. 1.
]

""" TODO: perhaps it is prudent to eliminate this. but this damping is pretty small."""
# Nonconservative forces (damping)
# damping is calculated by equating the mechanical power lost in a joint
# to the electrical power disipated in due to back EMF across the armature
# of the corresponding motor.
_F = (q,qdot)->[0,-qdot[θ1_idx]*Ke^2/R,-qdot[θ2_idx]*Ke^2/R]

"""
Computes the anchor dynamics as qddot = f(q,qdot,u), returns qddot
"""
function dynamics(q::Vector{T},qdot::Vector{T},u::Vector{T},p::Designs.Params) where T<:Real
    ∇V = potential_gradient(q,p)
    DA = A_jacobian(q,p)
    DAp = A_jacobian_prime(q,qdot,p)
    F = _F(q,qdot)
    λ = (DA*Minv*DA')\(DA*Minv*(∇V-G*u-F)-DAp*qdot)
    qddot = Minv*(-∇V+G*u+F+DA'*λ)
    return qddot,λ
end

# state projection map for the hopping behavior
const P = [1.0 0.0 0.0]

function anchor_projection(q::Vector{T},p::Designs.Params) where T<:Real
    return [q[1]-p.l1-p.l2-0.03]
end

"""
Computes the template dynamics at the projection of (q,qdot)
"""
function template_dynamics(q::Vector{T},qdot::Vector{T}) where T<:Real
    ω = 5pi
    ζ = 0.01
    return -2ζ*ω*qdot - ω^2 * q - [9.81]
end

function minimum_norm_control(q::Vector{T},qdot::Vector{T},p::Designs.Params) where T<:Real
    P = jacobian(q->anchor_projection(q,p),q)
    target = template_dynamics(anchor_projection(q,p),P*qdot)
    ∇V = potential_gradient(q,p)
    DA = A_jacobian(q,p)
    DAp = A_jacobian_prime(q,qdot,p)
    F = _F(q,qdot)
    A = vcat(
        hcat(DA, zeros(size(DA,1),size(DA,1)), zeros(size(DA,1),size(G,2))),
        hcat(M,-DA',-G),
        hcat(zeros(size(P)), P*Minv*DA', P*Minv*G)
    )
    b = vcat(
        -DAp*qdot,
        -∇V+F,
        P*Minv*(∇V-F)+target
    )
    return (A\b)[n+size(DA,1)+1:end]
end

function integration_mesh(p::Designs.Params)
    m=20
    (min, max) = (p.l2-p.l1+.04,p.l2+p.l1+.02)
    return range(min,max,length=m+1)
end

function interval_midpoint(a::Vector,b::Vector)
    return (a+b)/2
end

function cell_volume(a::Vector,b::Vector)
    prod(b-a)
end

function coord_transform(r::T,p::Designs.Params) where T<:Real
    f(θ) = constraints(vcat(r,θ...),p)
    θ = nlsolve(f,Array{T}([pi/4,-pi/4]);ftol=1e-6).zero
    return vcat(r,θ...)
end

function control_cost(p::Designs.Params)
    T = eltype(p.l1)
    mesh = integration_mesh(p)
    cost = 0
    q = zeros(T,n)
    qdot = zeros(T,n)
    for i = 1:length(mesh)-1
            midpoint = T((mesh[i]+mesh[i+1])/2)
            q = coord_transform(midpoint,p)
            u = minimum_norm_control(q,qdot,p)
            cell_volume = mesh[i+1]-mesh[i]
            cost += norm(u)^2*cell_volume
    end
    return cost / abs(mesh[end]-mesh[1])
end

function cost(x::Vector{T}) where T<:Real
    p = Designs.unpack(x)
    return control_cost(p)
end

function cost_grad(x::Vector{T}) where T<:Real
    cfg = GradientConfig(cost,x,Chunk{14}())
    return gradient(cost,x,cfg)
end

function cost_hessian(x::Vector{T}, h::T) where T<:Real
    return FiniteDifferences.central_difference(cost_grad,x,h)
end

function passive_dynamics(q::Vector, qdot::Vector, p::Designs.Params)
    return dynamics(q,qdot,zeros(eltype(q),2),p)
end

function active_dynamics(q::Vector, qdot::Vector, p::Designs.Params)
    return dynamics(q,qdot,minimum_norm_control(q,qdot,p),p)
end

"""
Calculates perturbation vectors that prevent accumulation of constraint
error. Keeps constraint errors to o(dt^2).
"""
function constraint_stabilization(q::Vector, qdot::Vector, p::Designs.Params)
    A = constraints(q,p)
    DA = A_jacobian(q,p)
    μ = -DA\A
    λ = -DA\(DA*qdot)
    return μ, λ
end

"""
Simulates the dynamics qddot = f(q,qdot) with timestep dt for N steps.
Incorporates constraint stabilization perturbations.
"""
function sim_euler(q0, qdot0, dt, N, p)
    q = zeros(eltype(q0), (length(q0),N+1))
    q[:,1] = q0
    qdot = zeros(eltype(qdot0), (length(qdot0),N+1))
    u = zeros(eltype(q0), (2,N+1))
    λ = zeros(eltype(q0), (2,N+1))
    qdot[:,1] = qdot0
    for i=1:N
        u[:,i] = minimum_norm_control(q[:,i],qdot[:,i],p)
        qddot,_λ = dynamics(q[:,i],qdot[:,i],u[:,i],p)
        λ[:,i] = _λ

        μ, ν = constraint_stabilization(q[:,i],qdot[:,i],p)
        q[:,i+1] = q[:,i]+dt*qdot[:,i]+μ
        qdot[:,i+1] = qdot[:,i]+dt*qddot+ν
    end
    u[:,end] = minimum_norm_control(q[:,end],qdot[:,end],p)
    return (q=q,qdot=qdot,u=u,λ=λ)
end

"""
What I need to do: take a hopper, simulate a collection of trajectories, compute the average energy expended
during those trajectories, and return the result.
"""

function sim_experiment(p)
    y0 = integration_mesh(p)
    cost = []
    for y in y0
        q0 = coord_transform(y,p)
        qdot0 = zeros(3)
        q,qdot,u,λ = sim_euler(q0,qdot0,1e-3,400,p)
        push!(cost, sum(u.*u)/size(u,2))
    end
    return cost
end

function alternate_control_cost(p::Designs.Params,tf)
    T = eltype(p.l1)
    mesh = integration_mesh(p)
    cost = 0
    q = zeros(T,n)
    qdot = zeros(T,n)
    τ = 1 / 4pi
    Δt = min(τ/10,tf/10)
    N = Int(floor(tf/Δt))
    for i = 1:length(mesh)-1
            cell_volume = mesh[i+1]-mesh[i]
            midpoint = T((mesh[i]+mesh[i+1])/2)
            q = coord_transform(midpoint,p)
            qdot = zeros(T,size(q))
            for j = 1:N
                u = minimum_norm_control(q,qdot,p)
                cost += norm(u)^2*cell_volume/N
                qddot,_λ = dynamics(q,qdot,u,p)
                μ,ν = constraint_stabilization(q,qdot,p)
                q += Δt*qdot+μ
                qdot += Δt*qddot+ν
            end
    end
    return cost / abs(mesh[end]-mesh[1])
end

function alternate_cost(x::Vector{T},tf) where T<:Real
    p = Designs.unpack(x)
    return alternate_control_cost(p,tf)
end

function alternate_cost_grad(x::Vector{T},tf) where T<:Real
    f = x->alternate_cost(x,tf)
    cfg = GradientConfig(f,x,Chunk{14}())
    return gradient(f,x,cfg)
end

function alternate_cost_hessian(x::Vector{T}, tf, h::T) where T<:Real
    return FiniteDifferences.central_difference(x->alternate_cost_grad(x,tf),x,h)
end


end