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

    @testset "Generator/source calling conventions" begin
        u0 = [1.0, 0.5]
        p = (decay = 0.3,)
        Gp = [-0.3 0.0; 0.3 -0.1]

        # generator and source as (p, t). A generic 2-arg callable matches (p, t).
        prob_pt = ContinuousIPMProblem((p, t) -> [-p.decay 0.0; p.decay -0.1],
            domain, u0, (0.0, 1.0); p = p, source = (p, t) -> [0.0, 0.05])
        du = zeros(2)
        to_ode_problem(prob_pt).f(du, u0, p, 0.4)
        @test du ≈ Gp * u0 .+ [0.0, 0.05]

        # A generic 3-arg callable always resolves to (u, p, t); the (u, t, p)
        # convention is only reachable when (u, p, t) does not match, so restrict
        # the time argument's type to force it.
        gen_utp(u, t::Real, q) = [-q.decay 0.0; q.decay -0.1]
        prob_utp = ContinuousIPMProblem(gen_utp, domain, u0, (0.0, 1.0); p = p)
        fill!(du, 0.0)
        to_ode_problem(prob_utp).f(du, u0, p, 0.4)
        @test du ≈ Gp * u0

        # Likewise (t, p) is reachable only when (p, t) does not match.
        gen_tp(t::Real, q) = [-q.decay 0.0; q.decay -0.1]
        prob_tp = ContinuousIPMProblem(gen_tp, domain, u0, (0.0, 1.0); p = p)
        fill!(du, 0.0)
        to_ode_problem(prob_tp).f(du, u0, p, 0.4)
        @test du ≈ Gp * u0

        # unsupported generator / source signatures error at lowering time
        @test_throws ArgumentError to_ode_problem(
            ContinuousIPMProblem((a, b, c, d) -> zeros(2, 2), domain, u0, (0.0, 1.0)))
        @test_throws ArgumentError to_ode_problem(
            ContinuousIPMProblem([-0.1 0.0; 0.0 -0.1], domain, u0, (0.0, 1.0);
                source = (a, b, c, d) -> zeros(2)))

        # length mismatches surface as DimensionMismatch from the RHS
        bad_src = ContinuousIPMProblem([-0.1 0.0; 0.0 -0.1], domain, u0, (0.0, 1.0);
            source = (u, p, t) -> [0.0])
        @test_throws DimensionMismatch to_ode_problem(bad_src).f(zeros(2), u0, nothing, 0.0)
    end

    @testset "Delay operator calling conventions" begin
        G = [-0.5 0.0; 0.2 -0.1]
        u0 = [1.0, 0.5]
        history(p, t) = [2.0, 1.0]
        Adelay = [0.0 0.4; 0.0 0.0]

        # callable operator returning a matrix, multiplied by the lagged state
        delay_mat = DelayGeneratorTerm(1.0, (u, p, t) -> Adelay)
        prob_mat = DelayIPMProblem(G, [delay_mat], domain, u0, history, (0.0, 2.0))
        du = zeros(2)
        dde = to_dde_problem(prob_mat)
        dde.f(du, u0, dde.h, nothing, 0.5)
        @test du ≈ G * u0 .+ Adelay * history(nothing, -0.5)

        # callable operator returning a vector contribution directly
        delay_vec = DelayGeneratorTerm(1.0, (u, h, p, t, lag) -> [0.3, -0.2])
        prob_vec = DelayIPMProblem(G, [delay_vec], domain, u0, history, (0.0, 2.0))
        fill!(du, 0.0)
        dde2 = to_dde_problem(prob_vec)
        dde2.f(du, u0, dde2.h, nothing, 0.5)
        @test du ≈ G * u0 .+ [0.3, -0.2]
    end

    @testset "PSPM scalar velocity field" begin
        transport_domain = ContinuousDomain(0.0, 1.0, 3)
        n0 = [1.0, 2.0, 3.0]
        prob = PSPMIPMProblem(transport_domain, n0, (0.0, 1.0); velocity = 1.0)
        du = zeros(3)
        to_ode_problem(prob).f(du, n0, nothing, 0.0)
        @test du ≈ [-3.0, -3.0, -3.0]
    end

    @testset "PSPM per-point and boundary-flux conventions" begin
        d = ContinuousDomain(0.0, 1.0, 3)
        zz = meshpoints(d)
        n0 = [1.0, 1.0, 1.0]
        hh = step_size(d)

        # A scalar-location mortality forces the per-point broadcast fallback,
        # since it is not applicable to the whole mesh vector.
        mort_pp(x::Real) = 0.1 * x
        prob = PSPMIPMProblem(d, n0, (0.0, 1.0); velocity = 0.0, mortality = mort_pp)
        du = zeros(3)
        to_ode_problem(prob).f(du, n0, nothing, 0.0)
        @test du ≈ -0.1 .* zz .* n0

        # boundary_lower as a (p, t) callable: constant inflow at the left face
        prob_bf = PSPMIPMProblem(d, n0, (0.0, 1.0); velocity = 1.0,
            boundary_lower = (p, t) -> 2.0)
        fill!(du, 0.0)
        to_ode_problem(prob_bf).f(du, n0, nothing, 0.0)
        @test du ≈ [-(1.0 - 2.0) / hh, 0.0, 0.0]

        # unsupported field / flux signatures error at lowering time
        @test_throws ArgumentError to_ode_problem(
            PSPMIPMProblem(d, n0, (0.0, 1.0); velocity = (a, b, c, d, e, f, g) -> 0.0))
        @test_throws ArgumentError to_ode_problem(
            PSPMIPMProblem(d, n0, (0.0, 1.0); velocity = 1.0,
                boundary_lower = (a, b, c, d, e, f) -> 0.0))
    end

    @testset "One-shot field/flux evaluators (extension contract)" begin
        # CategoricalPopulationDynamics' PSPM extension aggregates additive
        # sub-fields by calling these one-shot evaluators; keep them working.
        C = ContinuousStatePopulationDynamics
        pts = [0.1, 0.4, 0.9]
        pop = [1.0, 2.0, 3.0]
        aux = Float64[]

        @test C._evaluate_spatial_field(0.5, pts, pop, aux, nothing, 0.0, Float64;
            field_name = "mortality") == fill(0.5, 3)
        @test C._evaluate_spatial_field((z, population, a, p, t) -> 2 .* z,
            pts, pop, aux, nothing, 0.0, Float64; field_name = "source") ≈ 2 .* pts
        @test C._evaluate_spatial_field(nothing, pts, pop, aux, nothing, 0.0, Float64;
            field_name = "source") == zeros(3)

        d = ContinuousDomain(0.0, 1.0, 3)
        @test C._evaluate_boundary_flux(0.7, pop, aux, nothing, 0.0, d, Float64;
            field_name = "boundary_lower") == 0.7
        @test C._evaluate_boundary_flux((population, a, p, t, dom) -> sum(population),
            pop, aux, nothing, 0.0, d, Float64; field_name = "boundary_lower") == 6.0
        @test C._evaluate_boundary_flux(nothing, pop, aux, nothing, 0.0, d, Float64;
            field_name = "boundary_upper") == 0.0

        # auxiliary-rhs one-shot evaluator (used by the PSPM aux aggregator)
        aux_state = [0.5, 1.0]
        @test C._evaluate_auxiliary_rhs(nothing, pop, aux_state, nothing, 0.0, d, Float64) ==
            zeros(2)
        @test C._evaluate_auxiliary_rhs(
            (population, a, p, t, dom) -> [sum(population), a[1]],
            pop, aux_state, nothing, 0.0, d, Float64) == [6.0, 0.5]
        @test C._evaluate_auxiliary_rhs(nothing, pop, Float64[], nothing, 0.0, d, Float64) ==
            Float64[]
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
