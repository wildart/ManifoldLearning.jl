using Documenter, ManifoldLearning

makedocs(
	modules = [ManifoldLearning],
	doctest = false,
	clean = true,
	sitename = "ManifoldLearning.jl",
	pages = [
		"Home" => "index.md",
		"Interface" => "interface.md",
		"Methods" => [
			"Isomap" => "isomap.md",
			"Locally Linear Embedding" => "lle.md",
		# 	"HLLE" => "hlle.md",
		# 	"LEM" => "lem.md",
		# 	"LTSA" => "ltsa.md",
			"Diffusion maps" => "diffmap.md",
		],
		"Misc" => [
			# "Nearest Neighbors" => "knn.md",
			"Datasets" => "datasets.md",
		],
	]
)

#deploydocs(repo = "github.com/wildart/ManifoldLearning.jl.git")
