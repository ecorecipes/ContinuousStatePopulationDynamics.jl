using Test
using ContinuousStatePopulationDynamics

@testset "ContinuousStatePopulationDynamics" begin
    include("test_continuous.jl")
    include("test_demographic.jl")
    include("test_pbdm_migrations.jl")
end
