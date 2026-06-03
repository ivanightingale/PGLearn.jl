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

case_name = "118_ieee"

network = make_basic_network(pglib(case_name))
groups = Set.(PGLearn.OPFData(network, compute_clique_decomposition=true).clique_decomposition)

# # PGLearn outputs
# solver = "clarabel"
# precision = 128
# if solver == "mosek"
#     res_vec = JLD2.load("cholesky_$(case_name)_msk.jld2", "data")
# elseif precision == 128
#     res_vec = JLD2.load("cholesky_$(case_name)_128.jld2", "data")
# elseif precision == 64
#     res_vec = JLD2.load("cholesky_$(case_name).jld2", "data")
# end
# load_sums, trius = reconstruct_trius_dict(res_vec, groups)
# trius_pred = nothing

# DCP outputs
n_instances = 400
pred_name = "nu_polar_lin_loads_fc_4_512_s_sl0.01_500_scale_0.001_2_2_512_512_512"
file_path = joinpath("data", case_name, "$(pred_name).h5")
load_sums, trius, trius_pred = load_dcp_trius(file_path, groups; n_instances)

groups = filter_groups(trius; k=30)

n_rows = ceil(Int, sqrt(length(groups)))
n_cols = ceil(Int, length(groups) / n_rows)
row_size = 600
legend_size = 200
fig = Figure(size=(row_size * n_cols, row_size * n_rows + legend_size))

palette = colorschemes[:tab10]
global axs = []
global sample_ax = nothing  # for adding legends
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
            load_sums,
            real(trius[group][i, i, :]);
            markersize=markersize,
            color=true_re_color
        )
        scatter!(
            ax,
            load_sums,
            imag(trius[group][i, i, :]);
            markersize=markersize,
            color=true_im_color
        )
        if !isnothing(trius_pred)
            scatter!(
                ax,
                load_sums,
                real(trius_pred[group][i, i, :]);
                markersize=markersize,
                color=pred_re_color
            )
            scatter!(
                ax,
                load_sums,
                imag(trius_pred[group][i, i, :]);
                markersize=markersize,
                color=pred_im_color
            )
        end
        push!(axs, ax)
        for j in i+1:n
            ax = Axis(group_grid[i, j])
            isnothing(sample_ax) && (global sample_ax = ax)
            scatter!(
                ax,
                load_sums,
                real(trius[group][i, j, :]);
                markersize=markersize,
                color=true_re_color,
                label="Re"
            )
            scatter!(
                ax,
                load_sums,
                imag(trius[group][i, j, :]);
                markersize=markersize,
                color=true_im_color,
                label="Im"
            )
            if !isnothing(trius_pred)
                scatter!(
                    ax,
                    load_sums,
                    real(trius_pred[group][i, j, :]);
                    markersize=markersize,
                    color=pred_re_color,
                    label="Re - $(pred_name)"
                )
                scatter!(
                    ax,
                    load_sums,
                    imag(trius_pred[group][i, j, :]);
                    markersize=markersize,
                    color=pred_im_color,
                    label="Im - $(pred_name)"
                )
            end
            push!(axs, ax)
        end
    end
    Label(group_grid[0, :], string(group), fontsize = 20)
    linkyaxes!(axs...)
    global axs = []
end
# linkyaxes!(axs...)

Legend(
    fig[end+1, :],
    sample_ax;
    orientation=:vertical,
    labelsize=20
)
fig.layout.rowsizes = vcat(repeat([Relative(row_size / (n_rows * row_size + legend_size))]; outer=n_rows), [Relative(legend_size / (n_rows * row_size + legend_size))])

# # PGLearn
# if solver == "mosek"
#     save("cholesky_plots_$(case_name)_msk.png", fig)
# elseif precision == 128
#     save("cholesky_plots_$(case_name)_128.png", fig)
# elseif precision == 64
#     save("cholesky_plots_$(case_name).png", fig)
# end

# DCP
save("cholesky_plots_$(case_name)_$(pred_name).png", fig)
