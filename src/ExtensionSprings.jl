module ExtensionSprings

using NLopt
using JuMP
using LinearAlgebra
using ForwardDiff

struct Spring{T<:Real}
    Na::T # number of active coils
    d::T  # wire diameter
    C::T  # spring
end

# all data is for 302 stainless steel and given in N/m^2
const G = 68.95e9 # shear modulus
const E = 193e9 # elastic modulus
const Ssy = 275e6 # yield strength
const Sut = 620e6# ultimate tensile strength
const Ssu = .67*Sut # torsional rupture strength
const Ssa = 241e6 # amplitude fatigue component (Zimmerli)
const Ssm = 379e6 # mean fatigue component (Zimmerli) 
const Sse = Ssa/(1-(Ssm/Ssu)^2) # shear undurance limit (Zimmerli)

function outer_diameter(s::Spring)
    mean_diameter = s.C*s.d
    return mean_diameter+s.d
end

function body_coils(s::Spring)
    s.Na-G/E
end

function free_length(s::Spring)
    s.d*(2s.C-1+body_coils(s))
end

function spring_rate(s::Spring)
    s.d*G/(8s.C^3*s.Na)
end

function initial_stress_bounds(s::Spring)
    conversion_factor = 6.89476e3 # formula is for PSI, factor converts to PA
    τ_min = conversion_factor*(33500/exp(.105*s.C)-1000*(4-s.C/6.5))
    τ_max= conversion_factor*(33500/exp(.105*s.C)+1000*(4-s.C/6.5))
    return τ_min,τ_max
end

function initial_tension_bounds(s::Spring)
    τ_min,τ_max = initial_stress_bounds(s)
    f_min = τ_min*(pi*s.d^2)/(8*s.C)
    f_max = τ_max*(pi*s.d^2)/(8*s.C)
    return f_min, f_max # these will be what?? results appear off by 10^3
end

function mean_initial_tension(s::Spring)
    f_min, f_max = initial_tension_bounds(s)
    return .5*(f_min+f_max)
end

function spring_force(s::Spring,Δx::T) where T<:Real
    Fi = mean_initial_tension(s)
    spring_rate(s)*Δx+Fi
end

function spring_energy(s::Spring,Δx::T) where T<:Real
    .5*spring_rate(s)*Δx^2+mean_initial_tension(s)*Δx
end

"""
Computes the Bergstrasser curvature correction factor,
used to get better approximations of shear stress as a function of spring displacement.
"""
function bergstrasser_factor(s::Spring)
    (4*s.C+2)/(4*s.C-3)
end

function shear_stress(s::Spring,Δx) # should convert these to MPA or GPA for scaling
    berg = bergstrasser_factor(s)
    Fi = mean_initial_tension(s)
    F = spring_rate(s)*Δx+Fi
    (berg*8*F*s.C)/(pi*s.d^2)
end

function yield_deflection(s::Spring,n)
    σ = Ssy/n
    berg = bergstrasser_factor(s)
    F = σ*(pi*s.d^2)/(berg*8*s.C)
    Fi = mean_initial_tension(s)
    (F-Fi)/spring_rate(s)
end

function goodman_criterion(σa,σm)
    (σa/Sse+σm/Ssu)
end

end