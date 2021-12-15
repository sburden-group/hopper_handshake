module PotentialEstimation

using LinearAlgebra
using ForwardDiff
using ..Designs
using ..Handshake

struct Parameters
    K1      # spring rate
    L1      # free length
    T1      # initial tension
    K2      # spring rate
    L2      # free length
    T2      # initial tension
    K3      # spring rate
    L3      # free length
end

function pack(p::Parameters)
    [p.K1,p.L1,p.T1,p.K2,p.L2,p.T2,p.K3,p.L3]
end

function unpack(x::Vector{T}) where T<:Real
    Parameters(x...)
end


function spring_energy(p::Params,θ1::T,θ2::T,design::Designs.Params) where {T <: Real}
    g = 9.81
    q = zeros(T,4)
    q[1:2] = [θ1,θ2]
    l = Handshake.leg_length(q,p)

    joint_location = Array{T}([design.l1*sin(θ1),-design.l1*cos(θ1)])
    r = (design.l1+p.L1+.015)
    fixed_end = Array{T}([r*sin(design.s1_r),-r*cos(design.s1_r)]) 
    free_end = joint_location + .015*(fixed_end-joint_location)/norm(fixed_end-joint_location)
    δx = fixed_end-free_end
    s1_energy = .5*p.K1*(p.L1-norm(δx))^2 + norm(δx)*p.T1

    joint_location = Array{T}([design.l1*sin(θ2),-design.l1*cos(θ2)])
    r = (design.l1+p.L2+.015)
    fixed_end = Array{T}([r*sin(design.s2_r),-r*cos(design.s2_r)]) 
    free_end = joint_location + .015*(fixed_end-joint_location)/norm(fixed_end-joint_location)
    δx = fixed_end-free_end
    s1_energy = .5*p.K2*(p.L2-norm(δx))^2 + norm(δx)*p.T2

    s3_energy = .5*p.K3*(p.L3-l)^2

    return T(s1_energy+s2_energy+s3_energy)
end

function spring_force(p::Params,θ1::T,θ2::T,design::Designs.Params) where {T <: Real}
    f(θ) = spring_energy(p,θ[1],θ[2],design)
    return ForwardDiff.gradient(f,[θ1,θ2])
end

end