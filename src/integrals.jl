module ECInts
export BasisSet, overlap, kinetic, nuclear, ERI_2e4c, ERI_2e3c, ERI_2e2c
try
  using GaussianBasis
  #using Lints # package which uses libint
catch
  println("GaussianBasis package not installed! Generation of integrals is not available.")
end

# TODO use GaussianBasis.read_basisset("cc-pvtz",atoms[2]) to specify non-default basis


end #module