using LinearAlgebra
using Combinatorics
using Quadmath
using Graphs
using GraphMakie
using CairoMakie
using PowerPlots
using Colors
using ColorSchemes
using Statistics
using JLD2

using PGLib
using PowerModels

using PGLearn

case_name = "30_ieee"
visualize_real_part = true
res_vec = JLD2.load("cholesky_$(case_name).jld2", "data")[1:32]
seeds = map(res -> res["input"]["seed"], res_vec)
load_sums = map(res -> sum(res["input"]["pd"] .+ res["input"]["qd"]), res_vec)
network = make_basic_network(pglib(case_name))
groups = PGLearn.OPFData(network, compute_clique_decomposition=true).clique_decomposition
pmg = PowerModelsGraph(network, node_components=[:bus], edge_components=[:branch], connected_components=[:hi])
g = Graphs.SimpleGraph(pmg.graph)

clique_sizes = length.(groups)
cg_vec = complete_graph.(clique_sizes)
cg = reduce(blockdiag, cg_vec)
g_map = reduce(vcat, groups)  # Vector of length nv(cg), mapping from cg index to g index

function upper_iterator(matrix)
    return (matrix[idx] for idx in CartesianIndices(matrix) if idx[1] < idx[2])
end

# construct Vector of Vectors storing the node/edge data of each problem instance
node_data = []
edge_data = []
for (i, res) in enumerate(res_vec)
    node_data_i_flat = []
    edge_data_i_flat = []
    for (j, group) in enumerate(groups)
        u = res["dual"]["cholesky"][group].U
        node_data_i_flat = vcat(node_data_i_flat, real.(diag(u)))
        edge_data_i_flat = vcat(edge_data_i_flat, visualize_real_part ? real.(collect(upper_iterator(u))) : imag.(collect(upper_iterator(u))))
    end
    push!(node_data, node_data_i_flat)
    push!(edge_data, edge_data_i_flat)
end

# dashed lines that connect nodes in groups that represent the same node in the original graph
cg_common_edges = map(
    v -> combinations(v, 2),
    [findall(==(i), g_map) for i in 1:nv(g)]
)

# create a consistent color scale
v_min, v_max = extrema(vcat(reduce(vcat, node_data), reduce(vcat, edge_data)))
cmap = :viridis
get_color = v -> get(colorschemes[cmap], Float64((v - v_min) / (v_max - v_min)))

layout = GraphMakie.Stress()

n_rows = 4
n_cols = ceil(Int, length(res_vec) / n_rows)
fig = Figure(size=(300 * n_cols, 300 * n_rows + 200))

for (plt_i, i) in enumerate(sortperm(load_sums))
    row = div(plt_i - 1, n_cols) + 1
    col = mod(plt_i - 1, n_cols) + 1
    ax = Axis(
        fig[row, col];
        title="Seed = $(seeds[i]), load = $(round(load_sums[i]; digits=2))"
        xticksvisible = false,
        yticksvisible = false,
        xticklabelsvisible = false,
        yticklabelsvisible = false,
        xgridvisible = false,
        ygridvisible = false
    )
    
    node_data_i_flat = node_data[i]
    edge_data_i_flat = edge_data[i]
    node_colors = get_color.(node_data_i_flat)
    edge_colors = get_color.(edge_data_i_flat)

    p = graphplot!(
        ax, cg;
        layout=layout,
        node_color=node_colors,
        edge_color=edge_colors,
        node_size=15,
        edge_width=2,
    )

    for e_vec in cg_common_edges
        for e in e_vec
            x1, y1 = p[:node_pos][][e[1]]
            x2, y2 = p[:node_pos][][e[2]]
            lines!(ax, [x1, x2], [y1, y2];
                color=:black,
                linestyle=:dash,
                linewidth=1.2,
                alpha=0.2
            )
        end
    end
end

Colorbar(fig[:, end+1], limits=(v_min, v_max), colormap=cmap, label="Diagonal (node) & $(visualize_real_part ? "real" : "imag") part of off-diagonal (edge)")

titlelayout = GridLayout(fig[0, 1], halign = :left, tellwidth = false)
Label(titlelayout[1, 1], "Cholesky factorization of $(case_name) solutions", halign = :left, fontsize = 30)
Label(titlelayout[2, 1], "(diagonal and $(visualize_real_part ? "real" : "imag") parts of off-diagonal entries)", halign = :left, fontsize = 26)
Label(titlelayout[3, 1], "Instances sorted in total load from top left to bottom right. Dashed lines connect variables that represent the same bus in the network.", halign = :left, fontsize=20)
rowgap!(titlelayout, 0)

# fig
save("cholesky_$(case_name)_$(visualize_real_part ? "real" : "imag").png", fig)
