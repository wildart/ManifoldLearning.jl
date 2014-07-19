my_tests = ["utils.jl",
            "isomap.jl",
            "lle.jl",
            "hlle.jl",
            "ltsa.jl",
            "lem.jl",
            "diffmaps.jl"]

println("Running tests:")

for my_test in my_tests
    println(" * $(my_test)")
    include(my_test)
end
