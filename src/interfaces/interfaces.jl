"""
Interfaces module

This module provides functions to import (export) matrices from (to) external programs.

See also: [MolproInterface](@ref)
"""
module Interfaces
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.MolproInterface: MolproInterface, is_matrop_file
using ..ElemCo.MoldenInterface: MoldenInterface, is_molden_file

export import_matrix, export_molden_orbitals

"""
    import_matrix(EC::ECInfo, filename::String)

  Import a matrix from an external program.

  The type of the matrix is determined automatically.
"""
function import_matrix(EC::ECInfo, filename::String)
  if (type = is_matrop_file(filename))[1]
    if type[2] == :ORBITALS
      return MolproInterface.import_orbitals(EC, filename)
    elseif type[2] == :OVERLAP
      return MolproInterface.import_overlap(EC, filename)
    else
      error("Type of matrop file $filename not recognized. Call the appropriate function directly.")
    end
  else
    error("Type of $filename not recognized. Call the appropriate function directly.")
  end
end

function export_matrix(EC::ECInfo, filename::String, mat::AbstractArray)
  error("export_matrix not implemented yet")
end

"""
    export_molden_orbitals(EC::ECInfo, filename::String)

  Export the current orbitals to a Molden file.
"""
function export_molden_orbitals(EC::ECInfo, filename::String)
  MoldenInterface.write_molden_orbitals(EC, filename)
end

end # module