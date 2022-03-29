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
using Trapz


"""
BEGIN MODEL DEFINITION, DYNAMICS, CONTROL
"""
const m = 2.0
const Jm = .5/2*(.087^2+.08^2) # rough approximation motor inertia from thick-walled cylinder model.
const foot_offset = 0.03
const MIN_KNEE_DIST = 0.04
const KNEE_SPRING_OFFSET = 0.015
const CENTER_SPRING_OFFSET = 0.01

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

# gravity
const g = 9.81

# template constants
const ω_stance = 5pi 
const ζ_stance = 0.01
const ω_flight = 5pi 
const ζ_flight = 1.0

# kinematics
function hip_foot_angle(q::Vector{T}) where {T <: Real}
    (q[θ1_idx]-q[θ2_idx])/2.0
end

function interior_leg_angle(q::Vector{T}) where {T <: Real}
    (q[θ1_idx]+q[θ2_idx])/2.0
end

function leg_length(ϕ::T, p::Designs.Params) where {T <: Real}
    p.l1*cos(ϕ)+sqrt(p.l2^2-(p.l1*sin(ϕ))^2)+.03
end

function potential_energy(q::Vector{T},p::Designs.Params) where {T <: Real}
    g = 9.81
    ϕ = interior_leg_angle(q)
    l = leg_length(ϕ,p)
    θ1 = q[θ1_idx]
    θ2 = q[θ2_idx]

    s1_fl = Designs.ExtensionSprings.free_length(p.s1)
    joint_location = Array{T}([p.l1*sin(θ1),-p.l1*cos(θ1)])
    r = (p.l1+s1_fl+KNEE_SPRING_OFFSET)
    fixed_end = Array{T}([r*sin(p.s1_r),-r*cos(p.s1_r)]) 
    free_end = joint_location + KNEE_SPRING_OFFSET*(fixed_end-joint_location)/norm(fixed_end-joint_location)
    δx = fixed_end-free_end
    s1_energy = Designs.ExtensionSprings.spring_energy(p.s1,norm(δx)-s1_fl)

    s2_fl = Designs.ExtensionSprings.free_length(p.s2)
    joint_location = Array{T}([-p.l1*sin(θ2),-p.l1*cos(θ2)])
    r = (p.l1+s2_fl+KNEE_SPRING_OFFSET)
    fixed_end = Array{T}([r*sin(p.s2_r),-r*cos(p.s2_r)]) 
    free_end = joint_location + KNEE_SPRING_OFFSET*(fixed_end-joint_location)/norm(fixed_end-joint_location)
    δx = fixed_end-free_end
    s2_energy = Designs.ExtensionSprings.spring_energy(p.s2,norm(δx)-s2_fl)

    s3_energy = Designs.CompressionSprings.spring_energy(p.s3,p.l1+p.l2+CENTER_SPRING_OFFSET-l)

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

function stance_constraints(q::Vector{T},p::Designs.Params) where T<:Real
    θ = hip_foot_angle(q)
    ϕ = interior_leg_angle(q)
    l = leg_length(ϕ,p)
    return [q[body_idx]-l*cos(θ), l*sin(θ)]
end

function stance_constraints_jac(q::Vector{T},p::Designs.Params) where T<:Real
    f(q) = stance_constraints(q,p)
    cfg = JacobianConfig(f, q, Chunk{n}())
    jacobian(f,q,cfg)
end

function stance_constraints_hess(q::Vector{T},p::Designs.Params) where T<:Real
    f(x) = stance_constraints_jac(x,p)
    cfg = JacobianConfig(f, q, Chunk{n}())
    jacobian(f,q,cfg)
end

# input force map ( is actually constant in this model )
const G = [
    0. 0.
    1. 0.
    0. 1.
]

# Nonconservative forces (damping)
# damping is calculated by equating the mechanical power lost in a joint
# to the electrical power disipated in due to back EMF across the armature
# of the corresponding motor.
_F = (q,qdot)->[0,-qdot[θ1_idx]*Ke^2/R,-qdot[θ2_idx]*Ke^2/R]

function stance_dynamics(q::Vector{T},qdot::Vector{T},u::Vector{T},p::Designs.Params) where T<:Real
    ∇V = potential_gradient(q,p)
    DA = stance_constraints_jac(q,p)
    m = size(DA,1)
    ddtDA = reshape(stance_constraints_hess(q,p)*qdot,size(DA))
    F = _F(q,qdot)
    # solve system of linear equations Ax=b where x = (qddot, λ)
    A = vcat(   hcat(M,DA'),
                hcat(DA,zeros(m,m))
    )
    b = vcat(
        -∇V+G*u+F,
        -ddtDA*qdot
    )
    x = A\b
    qddot = x[1:n]
    λ = x[n+1:end]
    return qddot,λ
end

function stance_anchor_projection(q::Vector{T},p::Designs.Params) where T<:Real
    return [q[1]-p.l1-p.l2-foot_offset]
end

function stance_anchor_pushforward(q::Vector{T},p::Designs.Params) where T<:Real
    return [1.0 0.0 0.0]
end

function stance_template_dynamics(q::Vector{T},qdot::Vector{T}) where T<:Real
    return - ω_stance^2 * q - 2ζ_stance * ω_stance * qdot - [g]
end

function stance_control(q::Vector{T},qdot::Vector{T},p::Designs.Params) where T<:Real
    ∇V = potential_gradient(q,p)
    DA = stance_constraints_jac(q,p)
    m = size(DA,1)
    ddtDA = reshape(stance_constraints_hess(q,p)*qdot,size(DA))
    dπ = stance_anchor_pushforward(q,p) 
    f = stance_template_dynamics(stance_anchor_projection(q,p),dπ*qdot)
    F = _F(q,qdot)
    A = vcat(   hcat(M,DA',-G),
                hcat(DA,zeros(m,m),zeros(m,size(G,2))),
                hcat(dπ, zeros(size(dπ,1),m), zeros(size(dπ,1),size(G,2)))
    )
    b = vcat(
        -∇V+F,
        -ddtDA*qdot,
        f
    )
    return (A\b)[end-1:end]
end

function flight_dynamics(q::Vector{T},qdot::Vector{T},u::Vector{T},p::Designs.Params) where T<:Real
    ∇V = potential_gradient(q,p)
    F = _F(q,qdot)
    # in this particular model there are no constraints on flight dynamics
    qddot = Minv*(-∇V+G*u+F)
    return qddot
end

function flight_anchor_projection(q::Vector{T},p::Designs.Params) where T<:Real
    return q[[2,3]]
end

function flight_anchor_pushforward(q::Vector{T},p::Designs.Params) where T<:Real
    return [0. 1.0 0.0
            0.0 0.0 1.0]
end

function flight_template_dynamics(q::Vector{T},qdot::Vector{T}) where T<:Real
    return - ω_flight^2 * q - 2ζ_flight * ω_flight * qdot
end

function flight_control(q::Vector{T},qdot::Vector{T},p::Designs.Params) where T<:Real
    ∇V = potential_gradient(q,p)
    F = _F(q,qdot)
    dπ = flight_anchor_pushforward(q,p) 
    f = flight_template_dynamics(flight_anchor_projection(q,p),dπ*qdot)
    A = vcat(
        hcat(M,-G),
        hcat(dπ, zeros(size(dπ,1),size(G,2)))
    )
    b = vcat(
        -∇V+F,
        f
    )
    return (A\b)[end-1:end]
end


function integration_mesh()
    N = 5
    y0 = range(-0.13,-2g/ω_stance^2-.05,length=N)
    return y0
end

function template_trajs()
    y0 = integration_mesh()
    N = length(y0)
    A_stance = [0. 1. 0.
        -ω_stance^2 0. -1.
        0. 0. 0.]
    stance_flow = (x,t)->(exp(A_stance*t)*x)[[1,2]]
    τ_guess = (2pi/ω_stance)/4
    X_stance = zeros((N,2,2N))
    t_stance = zeros((N,2N))
    t_flight = zeros((N,2))
    for i=1:N
        x0 = [y0[i],0.,g]
        τ = nlsolve(t->[stance_flow(x0,t[1])[1]+.01],[τ_guess]).zero[1]
        x_τ = stance_flow(x0,τ) 
        T = 2τ + 2(x_τ[2]/g)
        t_stance[i,:] = Array(range(-τ,τ,length=2N))
        X_stance[i,:,:] = reduce(hcat,map(t->stance_flow(x0,t),t_stance[i,:]))
        t_flight[i,:] = [τ,T]
    end
    return (stance_state = X_stance, t_stance = t_stance, t_flight = t_flight)
end

# compute this data and store it
template_trajectories = template_trajs()

function template_immersion(y::T, ydot::T, p::Designs.Params) where T<:Real
    f = (q)->vcat(stance_constraints(q,p),stance_anchor_projection(q,p)-[y])
    df = (q)->vcat(stance_constraints_jac(q,p),stance_anchor_pushforward(q,p))
    q_guess = [leg_length(pi/2,p),pi/2,pi/2]
    q = nlsolve(f,df,q_guess).zero
    DA = stance_constraints_jac(q,p)
    Dπ = stance_anchor_pushforward(q,p)
    qdot = vcat(DA,Dπ)\vcat(zeros(size(DA,1)),ydot)
    return q, qdot
end

function trajectory_cost(idx::Int, p::Designs.Params)
    # get trajectory information
    stance_state = template_trajectories.stance_state[idx,:,:]
    t_stance = template_trajectories.t_stance[idx,:]
    t_flight = template_trajectories.t_flight[idx,:]
    # calculate control cost during stance
    thermal_loss = zeros(eltype(p.l1),length(t_stance))
    for i=1:length(t_stance)
        q,qdot = template_immersion(stance_state[:,i]...,p) 
        u = stance_control(q,qdot,p)
        thermal_loss[i] = sum(R*(u/Ke).^2)
    end
    stance_cost = trapz(t_stance,thermal_loss)
    # calculate control cost during flight
    y = p.l1+p.l2+foot_offset-.01
    q,qdot = template_immersion(y,typeof(y)(0.),p)
    u = flight_control(q,qdot,p)
    flight_cost = sum(R*(u/Ke).^2)*(t_flight[end]-t_flight[1])
    return stance_cost+flight_cost
end

function control_cost(p::Designs.Params)
    y0 = integration_mesh()
    N = length(y0)
    costs = [trajectory_cost(i,p) for i=1:N]
    cost = trapz(y0,costs)/abs(y0[end]-y0[1])
end

function smooth_abs(x::T,α::Float64) where T<:Real
    abs(x)+ 2/α * log(1+exp(-α*abs(x)))-2*log(2)/α
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
    A = stance_constraints(q,p)
    DA = stance_constraints_jac(q,p)
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
        u[:,i] = stance_control(q[:,i],qdot[:,i],p)
        qddot,_λ = stance_dynamics(q[:,i],qdot[:,i],u[:,i],p)
        λ[:,i] = _λ

        μ, ν = constraint_stabilization(q[:,i],qdot[:,i],p)
        q[:,i+1] = q[:,i]+dt*qdot[:,i]+μ
        qdot[:,i+1] = qdot[:,i]+dt*qddot+ν
    end
    u[:,end] = stance_control(q[:,end],qdot[:,end],p)
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

end