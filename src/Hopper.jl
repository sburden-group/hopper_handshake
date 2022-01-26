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
const m = 2.25
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
    p.l1*cos(ϕ)+sqrt(p.l2^2-(p.l1*sin(ϕ))^2)
end

function potential_energy(q::Vector{T},p::Designs.Params) where {T <: Real}
    g = 9.81
    return T(m*g*q[body_idx])
end

function kinetic_energy(q::Vector,qdot::Vector,p::Designs.Params)
    return 0.5*m*qdot[body_idx]^2+0.5*Jm*(qdot[θ1_idx]^2+qdot[θ2_idx]^2)
end

# mass matrix
const M = diagm([m,Jm,Jm])
const Minv = inv(M)

# coriolis matrix ( can actually neglect for now since it should be zero)
function coriolis_matrix(q::Vector,qdot::Vector,p::Designs.Params)
    x = vcat(q,qdot)
    f(x)=kinetic_energy(x[1:n],x[n+1:2n],p)
    Df(x)=gradient(f,x)
    cfg2=JacobianConfig(Df,x,Chunk{2n}())
    Y=jacobian(Df,x,cfg2)
    return Y[1:n,n+1:2n]+Y[n+1:2n,1:n]
end

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
Generates a random state from a random effector coordinates.
"""
function random_state(p::Designs.Params)
    q = zeros(n)
    qdot = zeros(n)
    q[body_idx] = body = 0.25+0.2*(rand()-.5)
    qdot[body_idx] = bodydot = 2*(rand()-.5)
    q0 = [pi/2,-pi/2];q0dot=[0.,0.];
    qj,qjdot = inverse_kin(q[[body_idx]], q0, p)
    q[[θ1_idx,θ2_idx]] = qj
    qdot[[θ1_idx,θ2_idx]] = qjdot
    return (q,qdot)
end

"""
Solves inverse kinematics problem given (pos,vel) of effector
"""
function inverse_kin(qe::Vector{T}, x0::Vector{T}, p::Designs.Params) where T<:Real
    f(x)=constraints(vcat(qe,x),p)
    Df(x)=A_jacobian(vcat(qe,x),p)[:,[θ1_idx,θ2_idx]]
    x = Array{T}(nlsolve(f,Df,Array{T}(x0)).zero)
    return x
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

# Nonconservative forces (damping)
# damping is calculated by equating the mechanical power lost in a joint
# to the electrical power disipated in due to back EMF across the armature
# of the corresponding motor.
_F = (q,qdot)->[0,-qdot[θ1_idx]*Ke^2/R,-qdot[θ2_idx]*Ke^2/R]


# state projection map for the hopping behavior
const P = [1.0 0.0 0.0]

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
    return qddot
end

"""
Computes the template dynamics at the projection of (q,qdot)
"""
function template_dynamics(q::Vector{T},qdot::Vector{T}) where T<:Real
    lt = 0.2 # location of minimum spring potential in template projection
    ω = 4pi
    ζ = 0.05
    kt = ω^2
    bt = 2ζ*ω
    qt = (P*q)[1] # scalarized because of stupid julia shit
    qtdot = (P*qdot)[1]
    return (-bt*qtdot-kt*(qt-lt))
end

"""
Computes the anchoring controller u which satisfies P*f(q,qdot,u) = g(P*q,P*qdot)
"""
function minimum_norm_control(q::Vector{T},qdot::Vector{T},p::Designs.Params) where T<:Real
    target = template_dynamics(q,qdot)
    ∇V = potential_gradient(q,p)
    DA = A_jacobian(q,p)
    DAp = A_jacobian_prime(q,qdot,p)
    F = _F(q,qdot)
    MinvDAT = Minv*DA'
    MinvG = Minv*G
    qddot = Minv*(∇V-F)
    A = vcat(
        hcat(DA*Minv*DA', DA*Minv*G),
        hcat(P*MinvDAT, P*MinvG)
    )
    b = vcat(
        DA*qddot-DAp*qdot,
        P*qddot+[target]
    )
    return (A\b)[3:4]
end

"""
Converts the 
"""
function coord_transform(x::Vector, p::Designs.Params)
    # converts integration coordinates to workspace coordinates
    T = eltype(p.k1) # y needs to be type of elements of p because p will be optimization variable
    y = zeros(T,6)
    y[2] = x[1]/2
    y[3] = -y[2]
    y[1] = leg_length(y[1:3],p)
    y[5] = x[2]/2
    y[6] = -y[5]
    DA = A_jacobian(y[1:3],p)
    y[4] = -(DA[1,2:3]'*y[5:6])/DA[1,1]
    return y
end

function control_cost(p::Designs.Params)
    T = eltype(p.l1)
    # Net for integration
    nrows = 10
    ncols = 3
    # This integration net is in polar coordinates
    (a1, b1) = (0.16, 0.24)            # bounds on leg length
    (a2, b2) = (-.2 , .2)              # bounds on leg angle
    net1 = range(a1,b1,length=nrows+1) # net over leg length
    net2 = range(a2,b2,length=ncols+1) # net over leg angle

    # helper functions for calculating sampling points and interval volume
    midpoint = i->i[1]+(i[2]-i[1])/2
    volume = I->(I[1][2]-I[1][1])*(I[2][2]-I[2][1]) # signed or not signed?

    # Calculations are done in this loop
    cost = 0
    q = zeros(T,n)
    qdot = zeros(T,n)
    qj = Array{T}([pi/4,-pi/4]) # initial guesses for inverse kinematics
    for i = 1:nrows

        for j = 1:ncols
            # calculate volume interval, integration point, and state
            I = ((net1[i],net1[i+1]),(net2[j],net2[j+1]))
            x = Array{T}([midpoint(I[1]),midpoint(I[2])])
            q[body_idx] = x[1]*cos(x[2]);
            θmean = x[2]
            θ1 = nlsolve(θ->leg_length([q[body_idx],θ[1]+θmean,2*θmean-θ[1]],p)-x[1], [θmean+pi/4];ftol=1e-6).zero[1]
            θ2 = 2*θmean-θ1
            q[[θ1_idx,θ2_idx]] = [θ1,θ2]
            u = minimum_norm_control(q,qdot,p)
            cost += norm(u)*abs(volume(I))
        end
    end
    # return cost
    # normalize by workspace volume
    return cost / (abs(b1-a1)*abs(b2-a2))
end

""" BEGIN OPTIMIZATION CODE """

function cost(x::Vector{T}) where T<:Real
    p = Designs.unpack(x)
    return control_cost(p)
end

function cost_grad(x::Vector{T}) where T<:Real
    cfg = GradientConfig(cost,x,Chunk{2}())
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

""" Demo of optimal passive vs. optimal active control vs. default active control """
function demo_plot(p::Designs.Params)
    N = 2000
    dt = .001
    t = Array(range(0,N*dt,length=N+1))

    # solve for initial conditions
    q0 = zeros(3); qdot0=zeros(3)
    q0[1] = 0.24
    f(x) = [q0[1]-leg_length([0.,x[1],-x[1]], p)]
    Df(x) = ForwardDiff.jacobian(f,x)
    q0[2] = nlsolve(f,Df,[pi/4]).zero[1]
    q0[3] = -q0[2]
    f1(q,qdot,p) = passive_dynamics(q,qdot,p)
    sim1 = sim_euler(f1, q0, qdot0, dt, N, p)
    f2(q,qdot,p) = active_dynamics(q,qdot,p)
    sim2 = sim_euler(f2, q0, qdot0, dt, N, p)

    # solve for initial conditions
    q0 = zeros(3); qdot0=zeros(3)
    q0[1] = 0.24
    g(x) = [q0[1]-leg_length([0,x[1],-x[1]], default_params)]
    Dg(x) = ForwardDiff.jacobian(g,x)
    q0[2] = nlsolve(g,Dg,[pi/2]).zero[1]
    q0[3] = -q0[2]
    sim3 = sim_euler(f2, q0, qdot0, dt, N, default_params)

    # compute controls for sim2 and sim3
    sim1_u_norm = zeros(N+1)
    sim2_u_norm = [
        norm(minimum_norm_control(sim2.q[:,i],sim2.qdot[:,i],p)) for i=1:N+1
    ]
    sim3_u_norm = [
        norm(minimum_norm_control(sim3.q[:,i],sim3.qdot[:,i],default_params)) for i=1:N+1
    ]

    # make plots
    trj_plot = plot(t, sim1.q[1,:],title="Position vs time.",label="passive optimized")
    plot!(trj_plot, t, sim2.q[1,:],label="active optimized")
    plot!(trj_plot, t, sim3.q[1,:],label="active unoptimized",linestyle=:dash)

    ctrl_plot = plot(t,sim1_u_norm,title="Control norm vs time.",label="")
    plot!(ctrl_plot,t,sim2_u_norm,label="")
    plot!(ctrl_plot,t,sim3_u_norm,label="",linestyle=:dash)
    plt=plot(trj_plot, ctrl_plot, layout=(2,1))
    return (plt, sim1, sim2, sim3)
end



end