module VCSEL_tests

  # load packages 
  using Random, LinearAlgebra, VCSEL, Test, StatsBase, Distributions

  # include test files 
  # include("vcselect_univariate_test.jl")
  # include("vcselect_interact_test.jl")
  # include("maxlambda_test.jl")
  # include("utilities_test.jl")
  
  include("VCSEL_test.jl")
  include("vcselect_multivariate_test.jl")
  include("linalgops_test.jl")
  include("genotype_kernel_test.jl")
  include("maxlambda_test.jl")
  
end # end of module