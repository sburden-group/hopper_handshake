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
const m = 1. # in this model, m is the payload mass at the effector
const Jm = .5/2*(.087^2+.08^2) # rough approximation motor inertia from thick-walled cylinder model.

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

# kinematics
function hip_foot_angle(q::Vector{T}) where {T <: Real}
    (q[θ1_idx]+q[θ2_idx])/2.0
end

function interior_leg_angle(q::Vector{T}) where {T <: Real}
    (q[θ1_idx]-q[θ2_idx])/2.0
end

function leg_length(q::Vector{T}, p::Designs.Params) where {T <: Real}
    ϕ = interior_leg_angle(q)
    p.l1*cos(ϕ)+sqrt(p.l2^2-(p.l1*sin(ϕ))^2)
end

# potential energy
function potential_energy(q::Vector{T},p::Designs.Params) where T <: Real
    g = 9.81
    return T(m*g*q[xf_idx])
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
function A_jacobian(q::Vector{T},p::Designs.Params) where T<:Real
    f = q->constraints(q,p)
    cfg = JacobianConfig(f, q, Chunk{n}())
    jacobian(f,q,cfg)
end

"""
Calculates the derivative of DA(q)
"""
function A_jacobian_prime(q::Vector{T},qdot::Vector{T},p::Designs.Params) where {T<:Real}
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
    DA = A_jacobian(q,p)
    DAp = A_jacobian_prime(q,qdot,p)
    F = _F(q,qdot)
    λ = (DA*Minv*DA')\(DA*Minv*(∇V-G*u-F)-DAp*qdot)
    qddot = Minv*(-∇V+G*u+F+DA'*λ)
    return qddot,λ
end

# state projection map for the handshake behavior
const P = [ 0.0 0.0 1.0 0.0
            0.0 0.0 0.0 1.0]

"""
Computes the template dynamics at the projection of (q,qdot)
"""
function template_dynamics(q::Vector{T},qdot::Vector{T}) where T<:Real
    lt = [0.0,0.2] # location of minimum spring potential in template projection
    ω = 2pi
    ζ = 0.5
    kt = ω^2
    bt = 2ζ*ω
    qt = (P*q)
    qtdot = (P*qdot)
    return (-bt*qtdot-kt*(qt-lt))
end

function minimum_norm_control(q::Vector{T},qdot::Vector{T},p::Designs.Params) where T<:Real
    target = template_dynamics(q,qdot)
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

function integration_mesh()
    nrows = 10
    ncols = 10
    # This integration net is in polar coordinates
    (amin, amax) = (.16, .24)              # bounds on leg length
    (bmin, bmax) = (-pi/4.0, pi/4.0)       # bounds on leg angle
    mesh1 = range(amin,amax,length=nrows+1) # net over leg length
    mesh2 = range(bmin,bmax,length=ncols+1) # net over leg angle
    return mesh1, mesh2
end


function coord_transform(r::T,θmean::T, p::Designs.Params) where T<:Real
    xf = r*cos(θmean)
    yf = r*sin(θmean)
    f(θ)=constraints(vcat(θ,xf,yf),p)
    θ = Array{T}(nlsolve(f,Array{T}([θmean+pi/4,θmean-pi/4]);ftol=1e-6).zero)
    return vcat(θ,xf,yf)
end

function control_cost(p::Designs.Params)
    T = eltype(p.l1)
    mesh1, mesh2 = integration_mesh()
    cost = 0.0
    q = zeros(T,n)
    qdot = zeros(T,n)
    for i = 1:length(mesh1)-1
        for j = 1:length(mesh2)-1
            a = [mesh1[i],mesh2[j]]
            b = [mesh1[i+1],mesh2[j+1]]
            midpoint = Array{T}((a+b)/2)
            q = coord_transform(midpoint..., p) 
            u = minimum_norm_control(q,qdot,p)
            cell_volume = abs(prod(b-a))
            cost += norm(u)*cell_volume
        end
    end
    a = [mesh1[1],mesh2[1]]; b = [mesh1[end],mesh2[end]];
    jointspace_volume = abs(prod(b-a))
    return cost / jointspace_volume
end

""" BEGIN OPTIMIZATION CODE """ 

function cost(x::Vector{T}) where {T <: Real}
    return control_cost(Designs.unpack(x))
end

function cost_grad(x::Vector{T}) where {T<:Real}
    cfg = GradientConfig(cost,x,Chunk{2}())
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

function sim_euler(f, q0, qdot0, dt, N, p)
    q = zeros(eltype(q0), (length(q0),N+1))
    q[:,1] = q0
    qdot = zeros(eltype(qdot0), (length(qdot0),N+1))
    qdot[:,1] = qdot0
    for i=1:N
        μ, λ = constraint_stabilization(q[:,i],qdot[:,i],p)
        qddot = f(q[:,i],qdot[:,i],p)
        q[:,i+1] = q[:,i]+dt*qdot[:,i]+μ
        qdot[:,i+1] = qdot[:,i]+dt*qddot+λ
    end
    return (q=q,qdot=qdot)
end

function sim_euler(f, q0, qdot0, dt, N, p)
    q = zeros(eltype(q0), (length(q0),N+1))
    q[:,1] = q0
    qdot = zeros(eltype(qdot0), (length(qdot0),N+1))
    qdot[:,1] = qdot0
    for i=1:N
        μ, λ = constraint_stabilization(q[:,i],qdot[:,i],p)
        qddot = f(q[:,i],qdot[:,i],p)
        q[:,i+1] = q[:,i]+dt*qdot[:,i]+μ
        qdot[:,i+1] = qdot[:,i]+dt*qddot+λ
    end
    return (q=q,qdot=qdot)
end

end