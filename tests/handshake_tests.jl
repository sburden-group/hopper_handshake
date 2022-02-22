using Revise
using Test
using DataFrames
using CSV
using ForwardDiff
# include("../HopperHandshake.jl") # reloading HopperHandshake.jl will trigger lots of recompilation

df = CSV.read("pareto_front.csv", DataFrame)
x = df[:,3:end]

@testset "handshake anchoring test" begin
    for i=1:size(x,2)
        p = Designs.unpack(x[:,i])
        mesh1,mesh2= Handshake.integration_mesh(p)
        for j=1:length(mesh1)-1
            for k=1:length(mesh2)-1
                    a = [mesh1[j],mesh2[k]]
                    b = [mesh1[j+1],mesh2[k+1]]
                    midpoint = (a+b)/2
                    q = Handshake.coord_transform(midpoint...,p)
                    qdot = zeros(4)
                    u = Handshake.minimum_norm_control(q,qdot,p)
                    qddot,λ = Handshake.dynamics(q,qdot,u,p)
                    P = ForwardDiff.jacobian(q->Handshake.anchor_projection(q,p),q)
                    @test P*qddot ≈ Handshake.template_dynamics(Handshake.anchor_projection(q,p),P*qdot)
                    @test sum(abs.(u)) < 60
            end
        end
    end
end