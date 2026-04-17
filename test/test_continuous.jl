import OrdinaryDiffEq

using SciMLBase: DDEProblem, ODEProblem

@testset "Continuous generator problems" begin
    domain = ContinuousDomain(0.0, 1.0, 2)

    @testset "ContinuousIPMProblem to ODEProblem" begin
        G = [-0.5 0.2;
              0.1 -0.3]
        u0 = [1.0, 0.5]
        prob = ContinuousIPMProblem(G, domain, u0, (0.0, 2.0); source = [0.1, 0.0])
        odeprob = to_ode_problem(prob)

        @test odeprob isa ODEProblem
        du = zeros(2)
        odeprob.f(du, odeprob.u0, odeprob.p, 0.0)
        @test du ≈ G * u0 .+ [0.1, 0.0]

        prob2 = remake(prob; tspan = (0.0, 3.0))
        @test prob2.tspan == (0.0, 3.0)
    end

    @testset "ContinuousIPMProblem with callable generator" begin
        u0 = [1.0, 0.5]
        prob = ContinuousIPMProblem(
            (u, p, t) -> [-p.decay * (1 + t) 0.0; p.decay 0.0],
            domain,
            u0,
            (0.0, 1.0);
            p = (decay = 0.2,),
            source = (u, p, t) -> [0.0, 0.05],
        )

        odeprob = to_ode_problem(prob)
        du = zeros(2)
        odeprob.f(du, odeprob.u0, odeprob.p, 0.5)
        expected = [-0.3 0.0; 0.2 0.0] * u0 .+ [0.0, 0.05]
        @test du ≈ expected
    end

    @testset "Continuous normalization" begin
        u0 = [1.0, 2.0]
        prob = ContinuousIPMProblem([0.2 0.0; 0.0 -0.1], domain, u0, (0.0, 1.0); normalize = true)
        odeprob = to_ode_problem(prob)
        du = zeros(2)
        odeprob.f(du, odeprob.u0, odeprob.p, 0.0)
        @test sum(du) ≈ 0.0 atol = 1e-10
    end

    @testset "DelayIPMProblem to DDEProblem" begin
        G = [-0.5 0.0;
              0.2 -0.1]
        delay = DelayGeneratorTerm(1.0, [0.0 0.4;
                                         0.0 0.0])
        history(p, t) = [2.0, 1.0]
        u0 = [1.0, 0.5]

        prob = DelayIPMProblem(G, [delay], domain, u0, history, (0.0, 2.0); source = [0.1, 0.0])
        ddeprob = to_dde_problem(prob)

        @test ddeprob isa DDEProblem
        du = zeros(2)
        ddeprob.f(du, ddeprob.u0, ddeprob.h, ddeprob.p, 0.5)
        @test du ≈ G * u0 .+ [0.0 0.4; 0.0 0.0] * history(nothing, -0.5) .+ [0.1, 0.0]

        prob2 = remake(prob; tspan = (0.0, 4.0))
        @test prob2.tspan == (0.0, 4.0)
    end

    @testset "PSPMIPMProblem to ODEProblem" begin
        transport_domain = ContinuousDomain(0.0, 1.0, 3)
        n0 = [1.0, 2.0, 3.0]
        prob = PSPMIPMProblem(transport_domain, n0, (0.0, 1.0);
            velocity = z -> 1.0,
            mortality = 0.0,
            boundary_lower = 0.0,
            source = 0.0)

        odeprob = to_ode_problem(prob)
        @test odeprob isa ODEProblem
        du = zeros(3)
        odeprob.f(du, odeprob.u0, odeprob.p, 0.0)
        @test du ≈ [-3.0, -3.0, -3.0]

        prob2 = remake(prob; tspan = (0.0, 2.0))
        @test prob2.tspan == (0.0, 2.0)
    end

    @testset "PSPMIPMProblem with auxiliary state" begin
        transport_domain = ContinuousDomain(0.0, 1.0, 2)
        prob = PSPMIPMProblem(transport_domain, [2.0, 1.0], (0.0, 1.0);
            velocity = 0.0,
            mortality = (z, population, aux, p, t) -> fill(aux[1], length(z)),
            source = (z, population, aux, p, t) -> ones(length(z)),
            aux0 = [0.5],
            auxiliary_rhs = (population, aux, p, t, domain) -> [sum(population) - aux[1]])

        odeprob = to_ode_problem(prob)
        du = zeros(3)
        odeprob.f(du, odeprob.u0, odeprob.p, 0.0)
        @test du ≈ [0.0, 0.5, 2.5]
    end

    @testset "Continuous solve dispatch errors" begin
        prob = ContinuousIPMProblem([-0.2 0.0; 0.0 -0.1], domain, [1.0, 1.0], (0.0, 1.0))
        dprob = DelayIPMProblem([-0.2 0.0; 0.0 -0.1], [DelayGeneratorTerm(1.0, zeros(2, 2))],
            domain, [1.0, 1.0], (p, t) -> [1.0, 1.0], (0.0, 1.0))
        pspmprob = PSPMIPMProblem(domain, [1.0, 1.0], (0.0, 1.0); velocity = 0.0)

        @test_throws ArgumentError solve(prob)
        @test_throws ArgumentError solve(prob, DirectIteration())
        @test_throws ArgumentError solve(dprob, EigenAnalysis())
        @test_throws ArgumentError solve(pspmprob, DirectIteration())
        @test_throws ArgumentError solve(pspmprob, EigenAnalysis())
    end

    @testset "Shared names remain available" begin
        @test SimpleIPM() isa AbstractIPMStructure
        @test SimpleContinuousState() isa AbstractContinuousStateStructure
        @test ContinuousState() isa AbstractStateSemantics
        @test DiscreteTime() isa AbstractTimeSemantics
    end
end
