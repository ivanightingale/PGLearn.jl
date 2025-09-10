using HDF5
using Colors
using Statistics:std

brighten(c::RGB, α=0.6) = RGB(
    clamp(c.r + α*(1 - c.r), 0, 1),
    clamp(c.g + α*(1 - c.g), 0, 1),
    clamp(c.b + α*(1 - c.b), 0, 1)
)

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
    # extract Dict(clique => (n_c × n_c × n_instances) Array) from a Vector of Dict of solutions
    res_vec_solved = filter((res) -> res["meta"]["termination_status"] in ["OPTIMAL", "SLOW_PROGRESS"], res_vec)
    triu_flat_vec = [res["dual"]["trius"] for res in res_vec_solved]  # Vector of Vectors with length (2 * total_trius_dim)
    trius_flat = reduce(hcat, triu_flat_vec)  # n_c × n_c × n_solved_instances
    load_sums = map(res -> sum(res["input"]["pd"] .+ res["input"]["qd"]), res_vec_solved)
    return load_sums, reconstruct_trius_dict(trius_flat, groups)
end

function filter_groups(trius::Dict; k=30)
    # if the number of cliques is too large, keep only the top k cliques with the greatest maximum standard deviation in its data
    # For each clique, compute the standard deviation of each entry across the instances and take the maximum
    return partialsort(collect(keys(trius)), 1:k; by=c -> maximum(std(trius[c], dims=3)), rev=true)
end

function load_hdf5_to_dict(file_path::String; n_instances=100)
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
