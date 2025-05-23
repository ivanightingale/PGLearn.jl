function test_opfdata(data::PGLearn.OPFData, network::Dict{String,Any})
    ref = PowerModels.build_ref(network)[:it][:pm][:nw][0]

    @test data.N == length(network["bus"])
    @test data.E == length(network["branch"])
    @test data.G == length(network["gen"])
    @test data.L == length(network["load"])

    # Bus data
    @test data.vmin == [ref[:bus][i]["vmin"] for i in 1:data.N]
    @test data.vmax == [ref[:bus][i]["vmax"] for i in 1:data.N]

    # Aggregate shunts at the bus level
    gs_ref = zeros(Float64, data.N)
    bs_ref = zeros(Float64, data.N)

    for (_, shunt) in ref[:shunt]  # filters out inactive shunts
        gs_ref[shunt["shunt_bus"]] += shunt["gs"]
        bs_ref[shunt["shunt_bus"]] += shunt["bs"]
    end

    @test data.gs == gs_ref

    # Aggregate loads at the bus level
    pd_ref = [network["load"]["$l"]["pd"] for l in 1:data.L]
    qd_ref = [network["load"]["$l"]["qd"] for l in 1:data.L]

    @test data.pd == pd_ref
    @test data.qd == qd_ref

    # Reference bus
    @assert length(ref[:ref_buses]) == 1
    @test data.ref_bus == first(keys(ref[:ref_buses]))

    # Generator data
    pgmin_ref = zeros(Float64, data.G)
    pgmax_ref = zeros(Float64, data.G)
    qgmin_ref = zeros(Float64, data.G)
    qgmax_ref = zeros(Float64, data.G)
    c0_ref = zeros(Float64, data.G)
    c1_ref = zeros(Float64, data.G)
    c2_ref = zeros(Float64, data.G)
    gen_status_ref = zeros(Int, data.G)
    bus_gens_ref = [Int[] for _ in 1:data.N]
    for g in 1:data.G  # does not filter out inactive generators
        gen = network["gen"]["$g"]
        pgmin_ref[g] = gen["pmin"]
        pgmax_ref[g] = gen["pmax"]
        qgmin_ref[g] = gen["qmin"]
        qgmax_ref[g] = gen["qmax"]
        c0_ref[g] = gen["cost"][3]
        c1_ref[g] = gen["cost"][2]
        c2_ref[g] = gen["cost"][1]
        gen_status_ref[g] = gen["gen_status"]
        push!(bus_gens_ref[gen["gen_bus"]], g)
    end

    sort!.(bus_gens_ref)

    @test data.pgmin == pgmin_ref
    @test data.pgmax == pgmax_ref
    @test data.qgmin == qgmin_ref
    @test data.qgmax == qgmax_ref

    @test data.c0 == c0_ref
    @test data.c1 == c1_ref
    @test data.c2 == c2_ref

    @test data.gen_status == (gen_status_ref .== 1)
    @test data.bus_gens == bus_gens_ref

    # Branch data
    bus_fr_ref = zeros(Int, data.E)
    bus_to_ref = zeros(Int, data.E)
    smax_ref = zeros(Float64, data.E)
    dvamin_ref = zeros(Float64, data.E)
    dvamax_ref = zeros(Float64, data.E)
    br_status_ref = zeros(Int, data.E)
    bus_arcs_fr_ref = [Int[] for _ in 1:data.N]
    bus_arcs_to_ref = [Int[] for _ in 1:data.N]
    for e in 1:data.E
        branch = network["branch"]["$e"]
        bus_fr_ref[e] = branch["f_bus"]
        bus_to_ref[e] = branch["t_bus"]
        smax_ref[e] = branch["rate_a"]
        dvamin_ref[e] = branch["angmin"]
        dvamax_ref[e] = branch["angmax"]
        br_status_ref[e] = branch["br_status"]
        push!(bus_arcs_fr_ref[bus_fr_ref[e]], e)
        push!(bus_arcs_to_ref[bus_to_ref[e]], e)
    end

    @test data.bus_fr == bus_fr_ref
    @test data.bus_to == bus_to_ref
    @test data.smax == smax_ref
    @test data.dvamin == dvamin_ref
    @test data.dvamax == dvamax_ref
    @test data.branch_status == (br_status_ref .== 1)
    @test data.bus_arcs_fr == bus_arcs_fr_ref
    @test data.bus_arcs_to == bus_arcs_to_ref

    if all(data.branch_status)
        @test data.A == PowerModels.calc_basic_incidence_matrix(network)
    end

    data_dict = PGLearn.to_dict(data)
    @test keys(data_dict["A"]) == Set(["I", "J", "V", "M", "N"])
    @test keys(data_dict["Ag"]) == Set(["I", "J", "V", "M", "N"])

    return nothing
end

function test_voltage_phasor_bounds(data::PGLearn.OPFData, network::Dict{String,Any})
    buspairs = PowerModels.calc_buspair_parameters(network["bus"], network["branch"])
    bp_wr_min, bp_wr_max, bp_wi_min, bp_wi_max = PowerModels.ref_calc_voltage_product_bounds(buspairs)

    wr_min, wr_max, wi_min, wi_max = PGLearn.compute_voltage_phasor_bounds(data)

    for e in 1:data.E
        if data.branch_status[e]
            bp = data.bus_fr[e], data.bus_to[e]
            @test wr_min[e] ≈ bp_wr_min[bp]
            @test wr_max[e] ≈ bp_wr_max[bp]
            @test wi_min[e] ≈ bp_wi_min[bp]
            @test wi_max[e] ≈ bp_wi_max[bp]
        end
    end

    return nothing
end

function test_voltage_phasor_bounds_scalar()
    vfmin = 0.9
    vfmax = 1.1
    vtmin = 0.9
    vtmax = 1.1
    
    # Symmetric bounds
    @testset "symmetric bounds" begin
        wrmin, wrmax, wimin, wimax = PGLearn.compute_voltage_phasor_bounds(vfmin, vfmax, vtmin, vtmax, -pi/3, pi/3)
        @test wrmin ≈ 0.81 * cos(pi/3)
        @test wrmax ≈ 1.21 * 1.0
        @test wimin ≈ -1.21 * sin(pi/3)
        @test wimax ≈ +1.21 * sin(pi/3)
    end

    # Non-symmetric bounds, but still different signs
    @testset "asynmetric bounds" begin
        wrmin, wrmax, wimin, wimax = PGLearn.compute_voltage_phasor_bounds(vfmin, vfmax, vtmin, vtmax, -pi/6, pi/3)
        @test wrmin ≈ 0.81 * cos(pi/3)
        @test wrmax ≈ 1.21
        @test wimin ≈ -1.21 * sin(pi/6)
        @test wimax ≈ +1.21 * sin(pi/3)
    end

    # Positive angles
    @testset "positive angles" begin
        wrmin, wrmax, wimin, wimax = PGLearn.compute_voltage_phasor_bounds(vfmin, vfmax, vtmin, vtmax, pi/6, pi/3)
        @test wrmin ≈ 0.81 * cos(pi/3)
        @test wrmax ≈ 1.21 * cos(pi/6)
        @test wimin ≈ 0.81 * sin(pi/6)
        @test wimax ≈ 1.21 * sin(pi/3)
    end

    # Negative angles
    @testset "negative angles" begin
        wrmin, wrmax, wimin, wimax = PGLearn.compute_voltage_phasor_bounds(vfmin, vfmax, vtmin, vtmax, -pi/3, -pi/4)
        @test wrmin ≈ 0.81 * cos(pi/3)
        @test wrmax ≈ 1.21 * cos(pi/4)
        @test wimin ≈ -1.21 * sin(pi/3)
        @test wimax ≈ -0.81 * sin(pi/4)
    end

    # Invalid bounds
    @testset "invalid bounds" begin
        @test_throws DomainError PGLearn.compute_voltage_phasor_bounds(vfmin, vfmax, vtmin, vtmax, -pi, 0.0)
        @test_throws DomainError PGLearn.compute_voltage_phasor_bounds(vfmin, vfmax, vtmin, vtmax, 0.0, pi/2 + 0.01)
    end

    return nothing
end
