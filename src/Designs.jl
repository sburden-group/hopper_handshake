module Designs
using ForwardDiff
using ..FiniteDifferences
using ..ExtensionSprings
using ..CompressionSprings
using LinearAlgebra: diagm
using Distributions

struct Params{T<:Real}
    s1::ExtensionSprings.Spring{T}
    s1_r::T
    s2::ExtensionSprings.Spring{T}
    s2_r::T
    s3::CompressionSprings.Spring{T}
    l1::T
    l2::T
end

const default_espring = ExtensionSprings.Spring(125.,1e-3,17.)
const default_cspring = CompressionSprings.Spring(80.,1e-3,20.,.4)
const default_params = Params(default_espring,pi/2,default_espring,-pi/2,default_cspring,0.12,0.17)


"""
The scale matrix is used to scale the decision vector of design parameters
so that all of the entries are about the same scale (~ 1).
"""
const scale_matrix = diagm([
        100             # active coils = 100x   (dimensionless)
        1e-3            # wire diameter = 1e-3x (convert millimeters to meters)
        10              # spring index = 10x    (dimensionless)
        1               # rest angle            (radians)
        100             # active coils = 100
        1e-3            #...
        10              #...
        1               #...
        100             # active coils = 100x
        1e-3            #...
        10              #...
        1e-1            # rest length of compression spring = .1x (convert decimeter to meter)
        1e-1            # proximal link length = .1x (convert decimeter to meter)
        1e-1            # distal link length = .1x
])

"""
Takes a Params struct, packs it into an array, and scales it accordingly.
"""
function pack(p::Params)
    x = [
        p.s1.Na         # spring 1 active coils
        p.s1.d          # spring 1 wire diameter
        p.s1.C          # spring 1 index
        p.s1_r          # spring 1 rest angle (joint space)
        p.s2.Na         # ...
        p.s2.d
        p.s2.C
        p.s2_r
        p.s3.Na         # spring 3 active coils
        p.s3.d          # spring 3 wire diamter
        p.s3.C          # spring 3 index
        p.s3.L0         # spring 3 rest length
        p.l1            # proximal link length
        p.l2            # distal link length
    ]
    scale_matrix\x
end
"""
Takes a vector of scaled design parameters, unscales them and
puts them into a Params struct.
"""
function unpack(x::Vector{T}) where T<:Real
    y = scale_matrix*x
    s1 = ExtensionSprings.Spring(y[1:3]...)
    s2 = ExtensionSprings.Spring(y[5:7]...)
    s3 = CompressionSprings.Spring(y[9:12]...)
    Params(s1,y[4],s2,y[8],s3,y[13:14]...)
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
        0.04        # lower bound on extension spring active coils, likely won't activate
        0.05         # lower bound on extension spring wire diameter, likely won't activate
        0.5         # lower bound on extension spring index, likely won't activate
        pi/2        # lower bound on right side extension spring rest angle
        0.04
        0.05
        0.5
        -3pi/4
        0.04        # lower bound on compression spring active coils
        0.05         # lower bound on compression spring wire diameter     
        0.5         # lower bound on compression spring index
        1.0         # lower bound on compression spring rest length, I hope this won't activate
        0.5         # lower bound on proximal link length
        1.0         # lower bound on distal link length         
    ];
    upper_bound = [
        1.5         # upper bound on extension spring active coils
        1.6         # upper bound on extension spring wire diameter
        1.5         # upper bound on extension spring index
        3pi/4       # upper bound on right side extension spring rest angle
        1.5
        1.6
        1.5
        -pi/2
        1.5         # upper bound on compression spring active coils, likely won't activate
        1.6         # upper bound on compression spring wire diameter, likely won't activate     
        1.5         # upper bound on compression spring index, likely won't activate
        4.0         # upper bound on compression spring rest length, I hope this won't activate
        1.0         # upper bound on proximal link length
        3.0         # upper bound on distal link length         
    ];
    
    return lower_bound, upper_bound
end

function spring_constraints(p::Params{T}) where T<:Real
    η = 1.2 # factor of safety for spring loading
    # σ1 = ExtensionSprings.shear_stress(p.s1,2*p.l1)/2
    # σ2 = ExtensionSprings.shear_stress(p.s2,2*p.l1)/2
    # minimum_compression = ΔL*exp(25ΔL)/(1+exp(25ΔL))
    # σ3 = CompressionSprings.shear_stress(p.s3,minimum_compression+2*p.l1)/2
    Array{T}([
        -ExtensionSprings.yield_deflection(p.s1,η)+2*p.l1
        -ExtensionSprings.yield_deflection(p.s2,η)+2*p.l1
        -CompressionSprings.yield_deflection(p.s3,η)+2*p.l1
        -CompressionSprings.maximum_deflection(p.s3)+2*p.l1
        CompressionSprings.outer_diameter(p.s3)-0.015
        p.s3.L0 - p.l1 - p.l2
        -ExtensionSprings.free_length(p.s1)+.08
        -ExtensionSprings.free_length(p.s2)+.08
        ExtensionSprings.free_length(p.s1)-.127
        ExtensionSprings.free_length(p.s2)-.127
    ])
end

function kinematic_constraints(p::Params{T}) where T<:Real
    Array{T}([
        -p.l1-p.l2+.27   # total link length lower bound
        p.l2-p.l1-.13    # minimum leg length upper bound
    ])
end

function nlconstraints(p::Params{T}) where T<:Real
    vcat(spring_constraints(p), kinematic_constraints(p))
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