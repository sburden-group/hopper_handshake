module Handshake

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
const m = 1.0 # in this model, m is the payload mass at the effector
const Jm = .5/2*(.087^2+.08^2) # rough approximation motor inertia from thick-walled cylinder model.
const foot_offset = 0.03
const MIN_KNEE_DIST = 0.04
const KNEE_SPRING_OFFSET = 0.015
const CENTER_SPRING_OFFSET = 0.01

# constants related to (unconstrained) statespace
const θ1_idx = 1
const θ2_idx = 2
const xf_idx = 3
const yf_idx = 4
const n = 4 # number of unconstrained configuration variables

# link inertias: for now, neglect entirely

# motor constants (am I including motor dynamics in the model?)
const R = 0.186
const Kv = (2pi*100.0/60) # (Rad/s) / Volt
const Ke = 1/Kv           # Volt / (rad/s)

# gravity
const g = 9.81

# template constants
const ω = 2pi
const ζ = 0.5

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

# potential energy
function potential_energy(q::Vector{T},p::Designs.Params) where T <: Real
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
    fixed_end = Array{T}([-r*sin(p.s2_r),-r*cos(p.s2_r)]) 
    free_end = joint_location + KNEE_SPRING_OFFSET*(fixed_end-joint_location)/norm(fixed_end-joint_location)
    δx = fixed_end-free_end
    s2_energy = Designs.ExtensionSprings.spring_energy(p.s2,norm(δx)-s2_fl)

    s3_energy = Designs.CompressionSprings.spring_energy(p.s3,p.l1+p.l2+CENTER_SPRING_OFFSET-l)

    return T(m*g*q[yf_idx]+s1_energy+s2_energy+s3_energy)
end

# kinetic energy
function kinetic_energy(q::Vector,qdot::Vector,p::Designs.Params)
    return 0.5*m*(qdot[xf_idx]^2+qdot[yf_idx]^2)+0.5*Jm*(qdot[θ1_idx]^2+qdot[θ2_idx]^2)
end

# mass matrix
const M = diagm([Jm,Jm,m,m])
const Minv = inv(M)

# ∇V for the model ( can simplify greatly since it should be representable as -Kx-F for fixed K,F in this model )
function potential_gradient(q::Vector{T},p::Designs.Params) where T<:Real
    f(x) = potential_energy(x,p)
    cfg = GradientConfig(f, q, Chunk{n}())
    return gradient(f,q,cfg)
end

# kinematic constraint on configuration due to foot contact at (0,0)
function constraints(q::Vector{T},p::Designs.Params) where T<:Real
    θ = hip_foot_angle(q)
    l = leg_length(q,p)
    return [q[xf_idx]-l*cos(θ),q[yf_idx]-l*sin(θ)]
end

# Constraint forces
function constraints_jac(q::Vector{T},p::Designs.Params) where T<:Real
    f = q->constraints(q,p)
    cfg = JacobianConfig(f, q, Chunk{n}())
    jacobian(f,q,cfg)
end

function constraints_hess(q::Vector{T},p::Designs.Params) where {T<:Real}
    rows = 2 # number of constraint equations
    f(x) = A_jacobian(x,p)
    cfg = JacobianConfig(f, q, Chunk{n}())
    reshape(jacobian(f,q,cfg)*qdot,rows,n)
end

# input force map ( is actually constant in this model )
const G = [
    1. 0.
    0. 1.
    0. 0.
    0. 0.
]

# Nonconservative forces (damping)
# damping is calculated by equating the mechanical power lost in a joint
# to the electrical power disipated in due to back EMF across the armature
# of the corresponding motor.
_F = (q,qdot)->[-qdot[θ1_idx]*Ke^2/R,-qdot[θ2_idx]*Ke^2/R,0,0]

"""
Computes the anchor dynamics as qddot = f(q,qdot,u), returns qddot
"""
function dynamics(q::Vector{T},qdot::Vector{T},u::Vector{T},p::Designs.Params) where T<:Real
    ∇V = potential_gradient(q,p)
    DA = constraints_jac(q,p)
    m = size(DA,1)
    ddtDA = reshape(constraints_hess(q,p)*qdot,size(DA))
    F = _F(q,qdot)
    A = vcat(
        hcat(M,DA'),
        hcat(DA,zeros(m,m))
    )
    b = vcat(
        -∇V+G*u+F,
        -ddtDA*qdot
    )
    x = A\b
    qddot = x[1:n]
    λ = x[n+1:end]
    return qddot, λ
end

function anchor_projection(q::Vector{T},p::Designs.Params) where T<:Real
    req = p.l2+p.l1+foot_offset-.01
    θeq = -pi/8
    return [q[3]-req*cos(θeq),q[4]-req*sin(θeq)]
end

function anchor_pushforward(q::Vector{T},p::Designs.Params) where T<:Real
    return [0. 0. 1.0 0.0
            0. 0. 0. 1.0]
end

"""
Computes the template dynamics at the projection of (q,qdot)
"""
function template_dynamics(q::Vector{T},qdot::Vector{T}) where T<:Real
    ω = 2pi
    ζ = 0.75
    return -2ζ*ω*qdot - ω^2 * q
end

function control(q::Vector{T},qdot::Vector{T},p::Designs.Params) where T<:Real
    ∇V = potential_gradient(q,p)
    DA = constraints_jac(q,p)
    m = size(DA,1)
    ddtDA = reshape(constraints_hess(q,p)*qdot,size(DA))
    dπ = anchor_pushforward(q,p)
    f = template_dynamics(anchor_projection(q,p),dπ*qdot)
    F = _F(q,qdot)
    A = vcat(
        hcat(M,DA',-G),
        hcat(DA,zeros(m,m),zeros(m,size(G,2))),
        hcat(dπ,zeros(size(dπ,1),m),zeros(size(dπ,1),size(G,2)))
    )
    b = vcat(
        -∇V+F
        -ddtDA*qdot
    )
    return (A\b)[end-1:end] 
end

function integration_mesh()
    N = 4
    M = 4
    r0 = range(

    )
    θ0 = range()
    return r0,θ0
end

function coord_transform(r::T,θ::T) where T<:Real
    translation = [-.12,0.]
    [r*cos(θ),r*sin(θ)]+translation
end

function coord_transform_jac(r::T,θ::T) where T<:Real
    jacobian(x->coord_transform(x...),[r,θ])
end

function template_trajs()
    r0,θ0 = integration_mesh()
    N = length(r0)
    M = length(θ0)
    A = [0. 0. 1. 0.
        0. 0. 0. 1.
        -ω^2 -2ζ*ω, 0. 0.
        0. 0. -ω^2 -2ζ*ω]
    flow = (x,t)->exp(A*t)*x
    state = zeros((N,M,size(A,1),2N*M))
    t = Array(range(0.,5/(ζ*ω),length=2N*M)) 
    for i=1:N
        for j=1:M
            x0 = coord_transform(r0[i],θ0[j])
            state[i,j,:,:] = reduce(hcat,map(t->flow(x0,t),t))
        end
    end
    return (state=state,t=t)
end

function template_immersion(x::Vector{T},xdot::Vector{T},p::Designs.Params) where T<:Real
    f = q->vcat(constraints(q,p),anchor_projection(q,p)-x)
    df = q->vcat(constraints_jac(q,p),anchor_purshforward(q,p))
    q_guess = [pi/4,pi/4,leg_length(pi/4,p),0.]
    q = nlsolve(f,df,q_guess).zero
    DA = constraints_jac(q,p)
    Dπ = anchor_pushforward(q,p)
    qdot = vcat(DA,Dπ)\vcat(zeros(size(DA,1)),xdot)
    return q,qdot
end

# function integration_mesh(p::Designs.Params)
#     nrows = 10
#     ncols = 10
#     # This integration net is in polar coordinates
#     r = p.l2+p.l1+.03
#     θ = -pi/8
#     (amin, amax) = (r-.06, r-.01)    # bounds on leg length
#     (bmin, bmax) = (θ-pi/8, θ+pi/8)        # bounds on leg angle
#     mesh1 = range(amin,amax,length=nrows+1) # net over leg length
#     mesh2 = range(bmin,bmax,length=ncols+1) # net over leg angle
#     return mesh1, mesh2
# end


function coord_transform(r::T,θmean::T, p::Designs.Params) where T<:Real
    xf = r*cos(θmean)
    yf = r*sin(θmean)
    f(θ)=constraints(vcat(θ,xf,yf),p)
    θ = Array{T}(nlsolve(f,Array{T}([θmean+pi/4,θmean-pi/4]);ftol=1e-6).zero)
    return vcat(θ,xf,yf)
end

function smooth_abs(x::T,α::Float64) where T<:Real
    abs(x)+ 2/α * log(1+exp(-α*abs(x)))-2*log(2)/α
end

function control_cost(p::Designs.Params)
    T = eltype(p.l1)
    mesh1, mesh2 = integration_mesh(p)
    cost = 0.0
    q = zeros(T,n)
    qdot = zeros(T,n)
    for i = 1:length(mesh1)-1
        for j = 1:length(mesh2)-1
            a = [mesh1[i],mesh2[j]]
            b = [mesh1[i+1],mesh2[j+1]]
            cell_volume = abs(prod(b-a))
            midpoint = Array{T}((a+b)/2)
            q = coord_transform(midpoint..., p) 
            u = minimum_norm_control(q,qdot,p)
            # cost += norm(u,2)*cell_volume
            # cost += sum(smooth_abs.(u,20.))*cell_volume
            cost += (R*norm(u/Ke,2)^2)*cell_volume
        end
    end
    a = [mesh1[1],mesh2[1]]; b = [mesh1[end],mesh2[end]];
    volume = abs(prod(b-a))
    return cost / volume
end

""" BEGIN OPTIMIZATION CODE """ 

function cost(x::Vector{T}) where {T <: Real}
    return control_cost(Designs.unpack(x))
end

function cost_grad(x::Vector{T}) where {T<:Real}
    cfg = GradientConfig(cost,x,Chunk{14}())
    return gradient(cost,x,cfg)
end

function cost_hessian(x::Vector{T}, h::T) where T<:Real
    return FiniteDifferences.central_difference(cost_grad,x,h)
end
""" SIMULATION CODE """

function passive_dynamics(q::Vector, qdot::Vector, p::Designs.Params)
    return dynamics(q,qdot,zeros(eltype(q),2),p)
end

function active_dynamics(q::Vector, qdot::Vector, p::Designs.Params)
    return dynamics(q,qdot,minimum_norm_control(q,qdot,p),p)
end

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
    r0,θ0 = integration_mesh(p)
    cost = []
    for r in r0
        for θ in θ0
            q0 = Handshake.coord_transform(r,θ,p)
            qdot0 = zeros(4)
            q,qdot,u,λ = sim_euler(q0,qdot0,1e-3,3000,p)
            unorm = [norm(u[:,i]) for i=1:size(u,2)]
            push!(cost, sum(unorm)/length(unorm))
        end
    end
    return cost
end

function alternate_control_cost(p::Designs.Params,tf)
    T = eltype(p.l1)
    mesh1,mesh2 = integration_mesh(p)
    cost = 0
    q = zeros(T,n)
    qdot = zeros(T,n)
    τ = 1 / 2pi
    Δt = min(tf/10, τ/10)
    N = Int(floor(tf/Δt))
    for i = 1:length(mesh1)-1
        for j = 1:length(mesh2)-1
            a = [mesh1[i],mesh2[j]]
            b = [mesh1[i+1],mesh2[j+1]]
            cell_volume = abs(prod(b-a))
            midpoint = Array{T}((a+b)/2)
            q = coord_transform(midpoint..., p) 
            qdot = zeros(T,size(q))
            for k = 1:N
                u = minimum_norm_control(q,qdot,p)
                cost += norm(u)^2*cell_volume/N
                qddot,_λ = dynamics(q,qdot,u,p)
                μ,ν = constraint_stabilization(q,qdot,p)
                q += Δt*qdot+μ
                qdot += Δt*qddot+ν
            end
        end
    end
    a = [mesh1[1],mesh2[1]]; b = [mesh1[end],mesh2[end]];
    volume = abs(prod(b-a))
    return cost / volume
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