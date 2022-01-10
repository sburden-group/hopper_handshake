module CompressionSprings


using NLopt
using JuMP
using LinearAlgebra
using ForwardDiff

struct Spring{T<:Real} 
    Na::T  # active coils
    d::T   # wire diameter
    C::T   # spring index
    L0::T  # free length
end

const G = 80e9 # shear modulus
const E = 210e9 # elastic modulus
const Ssy = 2170e6 # yield strength
const Ssu = .45*Ssy # torsional rupture strength
const Ssa = 241e6 # amplitude fatigue component (Zimmerli)
const Ssm = 379e6 # mean fatigue component (Zimmerli) 
const Sse = Ssa/(1-(Ssm/Ssu)^2) # shear endurance limit (Zimmerli)

function outer_diameter(s::Spring)
    mean_diameter = s.C*s.d
    return mean_diameter+s.d
end

function solid_height(s::Spring)
    # assuming both ends squared so that total coils = active coils + 2
    s.d*((s.Na+2)+1)
end

function pitch(s::Spring)
    (s.L0 - 3s.d)/s.Na
end

function spring_force(s::Spring,Δx::T) where T<:Real
    s.d*G*Δx/(8s.C^3*s.Na)
end

function spring_rate(s::Spring)
    spring_force(s,one(eltype(s.Na)))
end

function spring_energy(s::Spring,Δx::T) where T<:Real
    spring_force(s,Δx)*Δx
end

function bergstrasser_factor(s::Spring)
    (4*s.C+2)/(4*s.C-3)
end

function shear_stress(s::Spring,Δx::T) where T<:Real # should convert these to MPA or GPA for scaling
    berg = bergstrasser_factor(s)
    F = spring_force(s,Δx)
    (berg*8*F*s.C)/(pi*s.d^2)
end

function maximum_deflection(s::Spring) where T<:Real
    7*(s.L0-solid_height(s))/8 # maximum deflection for linear operation
end

function yield_deflection(s::Spring,n)
    σ = Ssy/n # maximum allowable stress
    berg = bergstrasser_factor(s)
    (σ*pi*s.d^2)/(berg*8*spring_rate(s)*s.C)
end

function maximum_load(s::Spring)
    spring_rate(s)*maximum_deflection(s)
end

function goodman_criterion(σa,σm)
    (σa/Sse+σm/Ssu)
end

end