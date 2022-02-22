using Revise
using Test
using DataFrames
using CSV
using ForwardDiff
# include("../HopperHandshake.jl") # reloading HopperHandshake.jl will trigger lots of recompilation

df = CSV.read("pareto_front.csv", DataFrame)
x = df[:,3:end]

@testset "hopper anchoring test" begin
    for i=1:size(x,2)
        p = Designs.unpack(x[:,i])
        mesh= Hopper.integration_mesh(p)
        for j=1:length(mesh)-1
                a = mesh[j]
                b = mesh[j+1]
                midpoint = (a+b)/2
                q = Hopper.coord_transform(midpoint,p)
                qdot = zeros(3)
                u = Hopper.minimum_norm_control(q,qdot,p)
                qddot,λ = Hopper.dynamics(q,qdot,u,p)
                P = ForwardDiff.jacobian(q->Hopper.anchor_projection(q,p),q)
                @test P*qddot ≈ Hopper.template_dynamics(Hopper.anchor_projection(q,p),P*qdot)
                @test λ[1] > 0
                @test abs(λ[2]) < λ[1]
                @test sum(abs.(u)) < 60
        end
    end
end
