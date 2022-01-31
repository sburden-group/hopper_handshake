using Revise
using Test
include("../HopperHandshake.jl") # reloading HopperHandshake.jl will trigger lots of recompilation

@testset "anchoring test" begin
    mesh1, mesh2 = Handshake.integration_mesh()
    for i=1:length(mesh1)-1
        for j=1:length(mesh2)-1
            p = Designs.unpack(Designs.random_sample(1)[:,1])
            a = [mesh1[i],mesh2[j]]
            b = [mesh1[i+1],mesh2[j+1]]
            midpoint = (a+b)/2
            qe = midpoint[1]*[sin(midpoint[2]),cos(midpoint[2])]
            θ = Handshake.coord_transform(qe, p) 
            q = vcat(θ,qe)
            qdot = zeros(4)
            u = Handshake.minimum_norm_control(q,qdot,p)
            qddot,λ = Handshake.dynamics(q,qdot,u,p)
            @test Handshake.P*qddot ≈ Handshake.template_dynamics(q,qdot)
        end
    end
end