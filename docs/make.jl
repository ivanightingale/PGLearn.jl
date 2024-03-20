using Documenter
using OPFGenerator

makedocs(
    sitename = "OPFGenerator",
    format = Documenter.HTML(),
    modules = [OPFGenerator]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
