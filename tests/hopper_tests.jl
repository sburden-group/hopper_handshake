using Revise
using Test
include("../HopperHandshake.jl") # reloading HopperHandshake.jl will trigger lots of recompilation

@testset "anchoring test" begin
    mesh1, mesh2 = Hopper.integration_mesh()
    for i=1:length(mesh1)-1
        for j=1:length(mesh2)-1
            p = Designs.unpack(Designs.random_sample(1)[:,1])
            a = [mesh1[i],mesh2[j]]
            b = [mesh1[i+1],mesh2[j+1]]
            midpoint = (a+b)/2
            q = Hopper.coord_transform(midpoint...,p)
            qdot = zeros(3)
            u = Hopper.minimum_norm_control(q,qdot,p)
            qddot,λ = Hopper.dynamics(q,qdot,u,p)
            @test Hopper.P*qddot ≈ Hopper.template_dynamics(q,qdot)
            @test λ[1] > 0
            @test abs(λ[2]) < λ[1]
        end
    end
end

@test Hopper.cost(Designs.random_sample(1)[:,1])