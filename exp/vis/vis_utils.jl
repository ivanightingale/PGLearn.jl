using HDF5
using Colors

brighten(c::RGB, α=0.6) = RGB(
    clamp(c.r + α*(1 - c.r), 0, 1),
    clamp(c.g + α*(1 - c.g), 0, 1),
    clamp(c.b + α*(1 - c.b), 0, 1)
)

function load_dict_hdf5(filename)
    d = Dict()
    h5open(filename, "r") do file
        # for each key which corresponds to a clique
        for key in keys(file)
            # convert string to Tuple then to Set, convert to Julia index
            parsed_key = ((key |> Meta.parse |> eval |> collect) .+ 1) |> Set
            d[parsed_key] = read(file[key])  # n_clique × n_clique × n_data
        end
    end
    return d
end
