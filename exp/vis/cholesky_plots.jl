using LinearAlgebra
using Combinatorics
using Quadmath
using Graphs
using CairoMakie
using PowerPlots
using ColorSchemes
using Statistics
using JLD2
using HDF5

using PGLib
using PowerModels

using PGLearn

include("vis_utils.jl")

case_name = "14_ieee"
pred_name = "nu_polar_lin_0.0001_2_512_512_512_cho"

res_vec = JLD2.load("cholesky_$(case_name).jld2", "data")
seeds = map(res -> res["input"]["seed"], res_vec)
load_sums = map(res -> sum(res["input"]["pd"] .+ res["input"]["qd"]), res_vec)
sorted_load_sums = sort(load_sums)
load_sums_perm = sortperm(load_sums)
pred_triu_dict = load_dict_hdf5(joinpath("data", case_name, "$(pred_name).h5"))

network = make_basic_network(pglib(case_name))
groups = Set.(PGLearn.OPFData(network, compute_clique_decomposition=true).clique_decomposition)

n_rows = ceil(Int, sqrt(length(groups)))
n_cols = ceil(Int, length(groups) / n_rows)
row_size = 600
legend_size = 200
fig = Figure(size=(row_size * n_cols, row_size * n_rows + legend_size))

palette = colorschemes[:tab10]
global axes = []
global sample_ax = nothing
markersize = 4
for (plt_i, group) in enumerate(groups)
    n = length(group)
    row = div(plt_i - 1, n_cols) + 1
    col = mod(plt_i - 1, n_cols) + 1
    group_grid = GridLayout(fig[row, col])
    true_re_color = palette[1]
    true_im_color = palette[2]
    pred_re_color = brighten(palette[1])
    pred_im_color = brighten(palette[2])
    for i in 1:n
        ax = Axis(group_grid[i, i])
        scatter!(
            ax,
            sorted_load_sums,
            [real(res["dual"]["cholesky"][group].U[i, i]) for res in res_vec][load_sums_perm];
            markersize=markersize,
            color=true_re_color
        )
        scatter!(
            ax,
            sorted_load_sums,
            real.(pred_triu_dict[group][i, i, eachindex(seeds)])[load_sums_perm];
            markersize=markersize,
            color=pred_re_color
        )
        push!(axes, ax)
        for j in i+1:n
            ax = Axis(group_grid[i, j])
            isnothing(sample_ax) && (global sample_ax = ax)
            scatter!(
                ax,
                sorted_load_sums,
                [real(res["dual"]["cholesky"][group].U[i, j]) for res in res_vec][load_sums_perm];
                markersize=markersize,
                color=true_re_color,
                label="Re"
            )
            scatter!(
                ax,
                sorted_load_sums,
                [imag(res["dual"]["cholesky"][group].U[i, j]) for res in res_vec][load_sums_perm];
                markersize=markersize,
                color=true_im_color,
                label="Im"
            )
            scatter!(
                ax,
                sorted_load_sums,
                real.(pred_triu_dict[group][i, j, eachindex(seeds)])[load_sums_perm];
                markersize=markersize,
                color=pred_re_color,
                label="Re - $(pred_name)"
            )
            scatter!(
                ax,
                sorted_load_sums,
                imag.(pred_triu_dict[group][i, j, eachindex(seeds)])[load_sums_perm];
                markersize=markersize,
                color=pred_im_color,
                label="Im - $(pred_name)"
            )
            push!(axes, ax)
        end
    end
    Label(group_grid[0, :], string(group), fontsize = 20)
    linkyaxes!(axes...)
    global axes = []
end

Legend(
    fig[end+1, :],
    sample_ax;
    orientation=:vertical,
    labelsize=20
)
fig.layout.rowsizes = vcat(repeat([Relative(row_size / (n_rows * row_size + legend_size))]; outer=n_rows), [Relative(legend_size / (n_rows * row_size + legend_size))])

save("cholesky_plots_$(case_name).png", fig)
