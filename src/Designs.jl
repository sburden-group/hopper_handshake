module Designs
using ForwardDiff
using ..FiniteDifferences
using ..ExtensionSprings
using ..CompressionSprings
using LinearAlgebra: diagm
using Distributions

struct Params{T<:Real}
    l1::T
    l2::T
    y_hop::T
    x_shake::T
    y_shake::T
end

const default_params = Params(0.12,0.17,0.1,0.1,0.)


"""
The scale matrix is used to scale the decision vector of design parameters
so that all of the entries are about the same scale (~ 1).
"""
const scale_matrix = diagm([
        1e-1            # proximal link length = .1x (convert decimeter to meter)
        1e-1            # distal link length = .1x
        1e-1
        1e-1
        1e-1
])

"""
Takes a Params struct, packs it into an array, and scales it accordingly.
"""
function pack(p::Params)
    x = [
        p.l1            # proximal link length
        p.l2            # distal link length
        p.y_hop
        p.x_shake
        p.y_shake
    ]
    scale_matrix\x
end
"""
Takes a vector of scaled design parameters, unscales them and
puts them into a Params struct.
"""
function unpack(x::Vector{T}) where T<:Real
    y = scale_matrix*x
    Params(y...)
end

"""
The constraints will no longer be conveniently represented in affine form.
This is because the new spring constraints will introduce nonlinearity.

How do I want to organize this code??

1) spring constraints
2) kinematics constraints
3) box constraints
"""
function bounds()
    lower_bound = [
        0.5         # lower bound on proximal link length
        1.0         # lower bound on distal link length
        1.0
        1.0
        -0.5         
    ];
    upper_bound = [
        1.0         # upper bound on proximal link length
        3.0         # upper bound on distal link length
        2.5
        2.5
        0.5   
    ];
    
    return lower_bound, upper_bound
end

function kinematic_constraints(p::Params{T}) where T<:Real
    Array{T}([
        -p.l1-p.l2+.25   # total link length lower bound
        -p.l2+p.l1+.1    # minimum leg length upper bound
        p.y_hop + .08 - p.l1 - p.l2
        -p.y_hop + .08 - p.l1 + p.l2
        p.x_shake^2+p.y_shake^2 - p.l1^2-p.l2^2
        p.y_shake/p.x_shake - tan(pi/4)
        -p.y_shake/p.x_shake - tan(pi/4)
    ])
end

function nlconstraints(p::Params{T}) where T<:Real
    kinematic_constraints(p)
end

function nlconstraints(x::Vector{T}) where T<:Real
    p = unpack(x)
    nlconstraints(p)
end

function nlconstraints_jacobian(x::Vector{T}) where T<:Real
    ForwardDiff.jacobian(nlconstraints,x)
end

function nlconstraints_hessian(x::Vector{T},h::T) where T<:Real
    return FiniteDifferences.central_difference(nlconstraints_jacobian,x,h)
end

function bound_constraints(x::Vector{T}) where T<:Real
    lb,ub = bounds()
    vcat(-x+lb,x-ub)
end

function bound_constraints(p::Params{T}) where T<:Real
    bound_constraints(pack(p))
end

function constraints(p::Params{T}) where T<:Real
    vcat(bound_constraints(p),nlconstraints(p))
end

function constraints(x::Vector{T}) where T<:Real
    vcat(bound_constraints(x),nlconstraints(x))
end

function constraints_jacobian(x::Vector{T}) where T<:Real
    In = diagm(ones(length(x)))
    return vcat(-In, In, nlconstraints_jacobian(x))
end

"""
Samples uniformly from the upper / lower bounds until N *feasible* designs
are found, returning said designs as an array of vectors
"""
function random_sample(N::Int)
    lb,ub = bounds()
    d = Uniform.(lb,ub) # vectorized
    x = zeros(length(lb),N)
    i = 1
    while i <= N
        sample = rand.(d) # vectorized
        if all(constraints(sample).<0)
            x[:,i] = sample[:]
            i+=1
        end
    end
    x
end

end # end of module Designs