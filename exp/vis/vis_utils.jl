using HDF5
using Colors
using Statistics:std

brighten(c::RGB, α=0.6) = RGB(
    clamp(c.r + α*(1 - c.r), 0, 1),
    clamp(c.g + α*(1 - c.g), 0, 1),
    clamp(c.b + α*(1 - c.b), 0, 1)
)

function _common_sparsity_pattern(mats::Vector{Hermitian{ComplexF64, SparseMatrixCSC{ComplexF64, Int64}}})
    # Get the set of (i,j) indices per matrix
    sets = [Set(zip(findnz(M)[1], findnz(M)[2])) for M in mats]
    
    # Compute intersection across all sets
    common = reduce(intersect, sets)
    
    I, J = collect(common) |> (x -> (first.(x), last.(x)))
    # Generate a sparse mask matrix
    return sparse(I, J, ones(length(I)), size(first(mats))...)
end

function reconstruct_S(s::Matrix, sr::Matrix, si::Matrix, buspair_fr::Vector, buspair_to::Vector)
    # convert (N × n_instances) s and (P × n_instances) sr and si data to (N × N × n_instances) S data
    N = size(s, 1)
    P = size(sr, 1)
    n_instances = size(s, 2)
    S_vec = [
        Hermitian(sparse(
            vcat(1:N, buspair_fr, buspair_to),
            vcat(1:N, buspair_to, buspair_fr),
            vcat(s[:, i], sr[:, i] + im * si[:, i], sr[:, i] - im * si[:, i])
        ))
        for i in 1:n_instances
    ]
    sparse_mask = _common_sparsity_pattern(S_vec)
    return stack(S_vec), sparse_mask
end

function reconstruct_S(res_vec::Vector, buspair_fr::Vector, buspair_to::Vector)
    # res_vec_solved = filter((res) -> res["meta"]["termination_status"] in ["OPTIMAL", "SLOW_PROGRESS"], res_vec)  # Clarabel
    res_vec_solved = filter((res) -> (res["meta"]["termination_status"] in ["OPTIMAL", "SLOW_PROGRESS"] && res["meta"]["dual_status"] == "FEASIBLE_POINT"), res_vec)  # Mosek

    s_flat_vec = [res["dual"]["s"] for res in res_vec_solved]
    s_flat = reduce(hcat, s_flat_vec)  # N × n_solved_instances
    sr_flat_vec = [res["dual"]["sr"] for res in res_vec_solved]
    sr_flat = reduce(hcat, sr_flat_vec)  # P × n_solved_instances
    si_flat_vec = [res["dual"]["si"] for res in res_vec_solved]
    si_flat = reduce(hcat, si_flat_vec)  # P × n_solved_instances
    load_sums = map(res -> sum(res["input"]["pd"] .+ res["input"]["qd"]), res_vec_solved)
    return load_sums, reconstruct_S(s, sr, si, buspair_fr, buspair_to)
end

function reconstruct_trius_dict(trius_flat::Matrix, groups::Vector)
    # convert ((2 * total_trius_dim) × n_instances) flat trius data to Dict(clique => (n_c × n_c × n_instances) Array)
    n_instances = size(trius_flat, 2)
    d = Dict()
    trius_r = trius_flat[1 : div(size(trius_flat, 1), 2), :]
    trius_i = trius_flat[div(size(trius_flat, 1), 2) + 1 : end, :]
    trius = trius_r + im .* trius_i  # total_trius_dim × n_instances
    trius_ptr = 1
    for c in groups
        n_c = length(c)
        triu_dim_c = div((n_c + 1) * n_c, 2)
        triu_mat = zeros(Complex{eltype(trius_flat)}, n_c, n_c, n_instances)
        # place the values into the matrix in row-major order
        for i in axes(triu_mat, 1)
            for j in i:size(triu_mat, 2)
                triu_mat[i, j, :] = trius[trius_ptr, :]
                trius_ptr += 1
            end
        end
        d[c] = triu_mat
    end
    return d
end

function reconstruct_trius_dict(res_vec::Vector, groups::Vector)
    # extract Dict(clique => (n_c × n_c × n_instances) Array) from a Vector of Dict of PGLearn solutions
    # res_vec_solved = filter((res) -> res["meta"]["termination_status"] in ["OPTIMAL", "SLOW_PROGRESS"], res_vec)  # Clarabel
    res_vec_solved = filter((res) -> (res["meta"]["termination_status"] in ["OPTIMAL", "SLOW_PROGRESS"] && res["meta"]["dual_status"] == "FEASIBLE_POINT"), res_vec)  # Mosek

    triu_flat_vec = [res["dual"]["trius"] for res in res_vec_solved]
    trius_flat = reduce(hcat, triu_flat_vec)  # (2 * total_trius_dim) × n_solved_instances
    load_sums = map(res -> sum(res["input"]["pd"] .+ res["input"]["qd"]), res_vec_solved)
    return load_sums, reconstruct_trius_dict(trius_flat, groups)
end

function filter_groups(trius::Dict; k=30)
    # if the number of cliques is too large, keep only the top k cliques with the greatest maximum standard deviation in its data
    # For each clique, compute the standard deviation of each entry across the instances and take the maximum
    return partialsort(collect(keys(trius)), 1:min(k, length(trius)); by=c -> maximum(std(trius[c], dims=3)), rev=true)
end

function filter_buses(S::Array; k=30)
    # if N is too large, keep only the top k buses with the greatest maximum magnitude in diag(S)
    # For each bus, compute the absolute mean of diag(S) across the instances and take the maximum
    N = size(S, 1)
    return partialsort(1:N, 1:min(k, N); by=i -> maximum(mean(abs.(S[i, i, :]), dims=3)), rev=true)
end

function load_dcp_S(file_path::String, buspair_fr::Vector, buspair_to::Vector; n_instances::Int=100)
    return h5open(file_path, "r") do file
        x_dataset = file["x"]
        s_dataset = file["nu_conic"]["s"]
        s_pred_dataset = file["nu_conic_pred"]["s"]
        sr_dataset = file["nu_conic"]["sr"]
        sr_pred_dataset = file["nu_conic_pred"]["sr"]
        si_dataset = file["nu_conic"]["si"]
        si_pred_dataset = file["nu_conic_pred"]["si"]
        instances_slice = 1:min(n_instances, size(trius_dataset, 2))
        load_sums = vec(sum(x_dataset["pd"][:, instances_slice] + x_dataset["qd"][:, instances_slice], dims=1))
        s = s_dataset[:, instances_slice]
        s_pred = s_pred_dataset[:, instances_slice]
        sr = sr_dataset[:, instances_slice]
        sr_pred = sr_pred_dataset[:, instances_slice]
        si = si_dataset[:, instances_slice]
        si_pred = si_pred_dataset[:, instances_slice]
        load_sums, reconstruct_S(s, sr, si, buspair_fr, buspair_to), reconstruct_S(s_pred, sr_pred, si_pred, buspair_fr, buspair_to)
    end
end

function load_dcp_trius(file_path::String, groups::Vector; n_instances::Int=100)
    return h5open(file_path, "r") do file
        x_dataset = file["x"]
        trius_dataset = file["nu_conic"]["trius"]
        trius_pred_dataset = file["nu_conic_pred"]["trius"]
        instances_slice = 1:min(n_instances, size(trius_dataset, 2))
        load_sums = vec(sum(x_dataset["pd"][:, instances_slice] + x_dataset["qd"][:, instances_slice], dims=1))
        trius = trius_dataset[:, instances_slice]
        trius_pred = trius_pred_dataset[:, instances_slice]
        load_sums, reconstruct_trius_dict(trius, groups), reconstruct_trius_dict(trius_pred, groups)
    end
end

function branches_to_buspairs(bus_fr, bus_to)
    # Given the branches, deduplicate parallel branches and get the lists of from buses and to buses
    # representing unique bus pairs
    # buspair_to_br: mapping from the index of each buspair to index of the first branch between that buspair
    # br_to_buspair: mapping from the index of each branch to the index of the corresponding buspair
    seen_buspairs = Dict()  # mapping from buspair tuple to index in the buspair_fr and buspair_to
    buspair_fr = []
    buspair_to = []
    buspair_to_br = []
    br_to_buspair = []
    
    for (i, (f, t)) in enumerate(zip(bus_fr, bus_to))
        if !((f, t) in seen_buspairs)
            seen_buspairs[(f, t)] = length(buspair_fr)  # current buspair index
            push!(buspair_fr, f)
            push!(buspair_to, t)
            push!(buspair_to_br, i)
        end
        push!(br_to_buspair, seen_buspairs[(f, t)])
    end

    return buspair_fr, buspair_to, buspair_to_br, br_to_buspair
