using JuMP
using PowerModels
using LinearAlgebra
mutable struct EconomicDispatch <: PM.AbstractPowerModel end

function build_opf(::Type{EconomicDispatch}, data::Dict{String,Any}, optimizer;
    T=Float64,
    soft_power_balance::Bool=false,
    soft_reserve_requirement::Bool=false,
    soft_thermal_limit::Bool=false,
    iterative_ptdf::Bool=true,
    power_balance_penalty=3500.0,
    reserve_shortage_penalty=1100.0,
    transmission_penalty=1500.0,
    minimum_reserve=0.0,
)
    # Cleanup and pre-process data
    PM.standardize_cost_terms!(data, order=2)
    PM.calc_thermal_limits!(data)
    ref = PM.build_ref(data)[:it][:pm][:nw][0]

    # Check that slack bus is unique
    length(ref[:ref_buses]) == 1 || error("Data has $(length(ref[:ref_buses])) slack buses (expected 1).")
    i0 = first(keys(ref[:ref_buses]))

    # Grab some data
    N = length(ref[:bus])
    G = length(ref[:gen])
    E = length(data["branch"])
    L = length(ref[:load])

    pmin = [ref[:gen][g][:pmin] for g in 1:G]
    pmax = [ref[:gen][g][:pmax] for g in 1:G]
    rmax = [ref[:gen][g][:rmax] for g in 1:G]
    PD = sum(ref[:load][l][:pd] for l in 1:L)
    pfmax = [data["branch"]["$e"]["rate_a"] for e in 1:E]

    model = JuMP.GenericModel{T}(optimizer)
    model.ext[:opf_model] = EconomicDispatch  # for internal checks
    model.ext[:opf_formulation] = Dict{Symbol,Any}(
        :soft_power_balance => soft_power_balance,
        :soft_reserve_requirement => soft_reserve_requirement,
        :soft_thermal_limit => soft_thermal_limit,
        :power_balance_penalty => power_balance_penalty,
        :reserve_shortage_penalty => reserve_shortage_penalty,
        :transmission_penalty => transmission_penalty,
        :minimum_reserve => minimum_reserve,
        :iterative_ptdf => iterative_ptdf,
    )

    # Variables

    JuMP.@variable(model, pmin[g] <= pg[g in 1:G] <= pmax[g])
    JuMP.@variable(model, 0 <= r[g in 1:G] <= rmax[g])
    JuMP.@variable(model, pf[e in 1:E])

    JuMP.@constraint(model,
        pf_lower_bound[e in 1:E],
        pf[e] >= -pfmax[e] - δf[e]
    )
    JuMP.@constraint(model,
        pf_upper_bound[e in 1:E],
        pf[e] <= pfmax[e] + δf[e]
    )

    JuMP.@variable(model, δpb_surplus >= 0)
    JuMP.@variable(model, δpb_shortage >= 0)
    JuMP.@variable(model, δr_shortage >= 0)
    JuMP.@variable(model, δf[1:E] >= 0)

    if !soft_power_balance
        JuMP.set_upper_bound(δpb_surplus, 0)
        JuMP.set_upper_bound(δpb_shortage, 0)
    end

    if !soft_reserve_requirement
        JuMP.set_upper_bound(δr_shortage, 0)
    end

    if !soft_thermal_limit
        JuMP.set_upper_bound.(δf, 0)
    end

    # Constraints

    JuMP.@constraint(model,
        total_generation[g in 1:G],
        pg[g] + r[g] <= pmax[g]
    )

    JuMP.@constraint(model,
        power_balance,
        sum(pg) + δpb_surplus - δpb_shortage == PD
    )

    JuMP.@constraint(model,
        reserve_requirement,
        sum(r) + δr_shortage >= minimum_reserve
    )

    model.ext[:PTDF] = PM.calc_basic_ptdf_matrix(data)
    model[:ptdf_flow] = Vector{JuMP.ConstraintRef}(undef, E)
    model.ext[:tracked_branches] = zeros(Bool, E)
    model.ext[:ptdf_iterations] = 0

    # Objective

    c = [ref[:gen][g][:cost][2] for g in 1:G]
    q = [get(ref[:gen][g], :reserve_cost, zeros(T, 3))[2] for g in 1:G]
    
    l, u = extrema(gen["cost"][1] for (i, gen) in ref[:gen])
    (l == u == 0.0) || @warn "Data $(data["name"]) has quadratic cost terms; those terms are being ignored"
    
    l, u = extrema(gen["reserve_cost"][1] for (i, gen) in ref[:gen])
    (l == u == 0.0) || @warn "Data $(data["name"]) has quadratic reserve cost terms; those terms are being ignored"

    JuMP.@objective(model, Min,
        sum(c[g]*pg[g] + ref[:gen][g][:cost][3] for g in 1:G) +
        sum(q[g]*r[g] + ref[:gen][g][:reserve_cost][3] for g in 1:G) +
        power_balance_penalty * δpb_surplus +
        power_balance_penalty * δpb_shortage +
        reserve_shortage_penalty * δr_shortage +
        transmission_penalty * sum(δf)
    )

    return OPFModel{EconomicDispatch}(data, model)
end

function solve!(opf::OPFModel{EconomicDispatch})
    data = opf.data
    model = opf.model
    T = typeof(model).parameters[1]
    
    if !model.ext[:opf_formulation][:iterative_ptdf]
        optimize!(opf.model, _differentiation_backend = MathOptSymbolicAD.DefaultBackend())
        return
    end

    N = length(data["bus"])
    G = length(data["gen"])
    E = length(data["branch"])
    L = length(data["load"])
    Al = sparse(
        [data["load"]["$l"]["load_bus"] for l in 1:L],
        [l for l in 1:L],
        ones(T, L),
        N,
        L,
    )

    Ag = sparse(
        [data["gen"]["$g"]["gen_bus"] for g in 1:G],
        [g for g in 1:G],
        ones(T, G),
        N,
        G,
    )

    pd = [data["load"]["$l"]["pd"] for l in 1:L]
    rate_a = [data["branch"]["$e"]["rate_a"] for e in 1:E]

    p_expr = Ag * model[:pg] - Al * pd

    solved = false
    niter = 0
    while !solved
        optimize!(opf.model, _differentiation_backend = MathOptSymbolicAD.DefaultBackend())
        st = termination_status(model)
        st == MOI.OPTIMAL || error("Solver failed to converge: $st")
        pg_ = value.(model[:pg])
        p_ = Ag * pg_ - Al * pd
        pf_ = model.ext[:PTDF] * p_

        n_violated = 0
        for e in 1:E
            if (model.ext[:tracked_branches][e]) || (-rate_a[e] <= pf_[e] <= rate_a[e])
                continue
            end

            model[:ptdf_flow][e] = JuMP.@constraint(
                model,
                model[:pf][e] == dot(model.ext[:PTDF][e, :], p_expr)
            )
            model.ext[:tracked_branches][e] = true
            n_violated += 1
        end
        solved = n_violated == 0
        niter += 1
    end

    @info("Solved in $niter iteration" * (niter == 1 ? "" : "s"))

    opf.model.ext[:ptdf_iter] = niter

    return
end

function extract_result(opf::OPFModel{EconomicDispatch})
    data = opf.data
    model = opf.model

    ref = PM.build_ref(data)[:it][:pm][:nw][0]
    N = length(ref[:bus])
    G = length(ref[:gen])
    E = length(data["branch"])

    res = Dict{String,Any}()
    res["opf_model"] = string(model.ext[:opf_model])
    res["objective"] = JuMP.objective_value(model)
    res["objective_lb"] = -Inf
    res["optimizer"] = JuMP.solver_name(model)
    res["solve_time"] = JuMP.solve_time(model)
    res["termination_status"] = string(JuMP.termination_status(model))
    res["primal_status"] = string(JuMP.primal_status(model))
    res["dual_status"] = string(JuMP.dual_status(model))
    res["solution"] = sol = Dict{String,Any}()

    sol["per_unit"] = get(data, "per_unit", false)
    sol["baseMVA"]  = get(data, "baseMVA", 100.0)

    sol["gen"] = Dict{String,Any}()
    for g in 1:G
        sol["gen"]["$g"] = Dict(
            "pg" => value(model[:pg][g]),
            "r" => value(model[:r][g]),

            "mu_pg" => dual(LowerBoundRef(model[:pg][g])) - dual(UpperBoundRef(model[:pg][g])),
            "mu_r" => dual(LowerBoundRef(model[:r][g])) - dual(UpperBoundRef(model[:r][g])),

            "mu_total_generation" => dual(model[:total_generation][g]),
        )
    end

    sol["branch"] = Dict{String,Any}()
    for e in 1:E
        if data["branch"]["$b"]["br_status"] == 0
            sol["branch"]["$e"] = Dict(
                "pf" => 0,
                "df" => 0,

                "mu_pf" => 0,
                "mu_df" => 0,
                "lam_ptdf" => 0,
            )
        else
            sol["branch"]["$e"] = Dict(
                "pf" => value(model[:pf][e]),
                "df" => value(model[:δf][e]),

                "mu_pf" => dual(model[:pf_lower_bound][e]) - dual(model[:pf_upper_bound][e]),
                "mu_df" => dual(LowerBoundRef(model[:δf][e])) - dual(UpperBoundRef(model[:δf][e])),
                "lam_ptdf" => isdefined(model[:ptdf_iter], b) ? dual(model[:ptdf_iter][b]) : 0.0
            )
        end
    end

    sol["singleton"] = Dict{String,Any}(
        "dpb_surplus" => value(model[:δpb_surplus]),
        "dpb_shortage" => value(model[:δpb_shortage]),
        "dr_shortage" => value(model[:δr_shortage]),

        "mu_dpb_surplus" => dual(LowerBoundRef(model[:δpb_surplus])) - dual(UpperBoundRef(model[:δpb_surplus])),
        "mu_dpb_shortage" => dual(LowerBoundRef(model[:δpb_shortage])) - dual(UpperBoundRef(model[:δpb_shortage])),
        "mu_dr_shortage" => dual(LowerBoundRef(model[:δr_shortage])) - dual(UpperBoundRef(model[:δr_shortage])),

        "mu_power_balance" => dual(model[:power_balance]),
        "mu_reserve_requirement" => dual(model[:reserve_requirement]),
    )

    return res
end

function json2h5(::Type{PM.EconomicDispatch}, res)
    sol = res["solution"]

    E = length(sol["branch"])
    G = length(sol["gen"])

    res_h5 = Dict{String,Any}(
        "meta" => Dict{String,Any}(
            "termination_status" => res["termination_status"],
            "primal_status" => res["primal_status"],
            "dual_status" => res["dual_status"],
            "solve_time" => res["solve_time"],
        ),
    )

    res_h5["primal"] = pres_h5 = Dict{String,Any}(
        "pg" => zeros(Float64, G),
        "r" => zeros(Float64, G),
        "pf" => zeros(Float64, E),

        "df" => zeros(Float64, E),
        "dpb_surplus" => 0
        "dpb_shortage" => 0
        "dr_shortage" => 0
    )

    res_h5["dual"] = dres_h5 = Dict{String,Any}(
        "mu_pg" => zeros(Float64, G),
        "mu_r" => zeros(Float64, G),
        "mu_total_generation" => zeros(Float64, G),
        "mu_pf" => zeros(Float64, E),
        "mu_df" => zeros(Float64, E),
        "lam_ptdf" => zeros(Float64, E),
        "mu_dpb_surplus" => 0,
        "mu_dpb_shortage" => 0,
        "mu_dr_shortage" => 0,
        "mu_power_balance" => 0,
        "mu_reserve_requirement" => 0,
    )

    for g in 1:G
        gsol = sol["gen"]["$g"]

        pres_h5["pg"][g] = gsol["pg"]
        pres_h5["r"][g] = gsol["r"]

        dres_h5["mu_pg"][g] = gsol["mu_pg"]
        dres_h5["mu_r"][g] = gsol["mu_r"]
        dres_h5["mu_total_generation"][g] = gsol["mu_total_generation"]
    end

    for e in 1:E
        brsol = sol["branch"]["$e"]

        pres_h5["pf"][e] = brsol["pf"]
        pres_h5["df"][e] = brsol["df"]

        dres_h5["mu_pf"][e] = brsol["mu_pf"]
        dres_h5["mu_df"][e] = brsol["mu_df"]
        dres_h5["lam_ptdf"][e] = brsol["lam_ptdf"]
    end

    pres_h5["dpb_surplus"] = sol["singleton"]["dpb_surplus"]
    pres_h5["dpb_shortage"] = sol["singleton"]["dpb_shortage"]
    pres_h5["dr_shortage"] = sol["singleton"]["dr_shortage"]

    dres_h5["mu_dpb_surplus"] = sol["singleton"]["mu_dpb_surplus"]
    dres_h5["mu_dpb_shortage"] = sol["singleton"]["mu_dpb_shortage"]
    dres_h5["mu_dr_shortage"] = sol["singleton"]["mu_dr_shortage"]
    dres_h5["mu_power_balance"] = sol["singleton"]["mu_power_balance"]
    dres_h5["mu_reserve_requirement"] = sol["singleton"]["mu_reserve_requirement"]

    return res_h5

end
