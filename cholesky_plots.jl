using LinearAlgebra
using Combinatorics
using Quadmath
using Graphs
using CairoMakie
using PowerPlots
using Colors
using ColorSchemes
using Statistics
using JLD2

using PGLib
using PowerModels

using PGLearn

case_name = "14_ieee"
res_vec = JLD2.load("cholesky_$(case_name).jld2", "data")[1:64]
seeds = map(res -> res["input"]["seed"], res_vec)
load_sums = map(res -> sum(res["input"]["pd"] .+ res["input"]["qd"]), res_vec)
sorted_load_sums = sort(load_sums)
load_sums_perm = sortperm(load_sums)
network = make_basic_network(pglib(case_name))
groups = PGLearn.OPFData(network, compute_clique_decomposition=true).clique_decomposition
pmg = PowerModelsGraph(network, node_components=[:bus], edge_components=[:branch], connected_components=[:hi])
g = Graphs.SimpleGraph(pmg.graph)

clique_sizes = length.(groups)
cg_vec = complete_graph.(clique_sizes)
cg = reduce(blockdiag, cg_vec)
g_map = reduce(vcat, groups)  # Vector of length nv(cg), mapping from cg index to g index

n_rows = ceil(Int, sqrt(length(groups)))
n_cols = ceil(Int, length(groups) / n_rows)
fig = Figure(size=(600 * n_cols, 600 * n_rows))

axes = []
markersize = 5
for (plt_i, group) in enumerate(groups)
    n = length(group)
    row = div(plt_i - 1, n_cols) + 1
    col = mod(plt_i - 1, n_cols) + 1
    group_grid = GridLayout(fig[row, col])
    for i in 1:n
        ax = Axis(group_grid[i, i])
        scatter!(
            ax,
            sorted_load_sums,
            [real.(res["dual"]["cholesky"][group].U[i, i]) for res in res_vec][load_sums_perm];
            markersize=markersize
        )
        push!(axes, ax)
        for j in i+1:n
            ax = Axis(group_grid[i, j])
            scatter!(
                ax,
                sorted_load_sums,
                [real.(res["dual"]["cholesky"][group].U[i, j]) for res in res_vec][load_sums_perm];
                markersize=markersize
            )
            scatter!(
                ax,
                sorted_load_sums,
                [imag.(res["dual"]["cholesky"][group].U[i, j]) for res in res_vec][load_sums_perm];
                markersize=markersize
            )
            push!(axes, ax)
        end
    end
    Label(group_grid[0, :], string(group), fontsize = 20)
end
linkyaxes!(axes...)

save("cholesky_plots_$(case_name).png", fig)
