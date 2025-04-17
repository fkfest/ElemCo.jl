module Wavefunctions
using HDF5
using ..ElemCo.AbstractEC
using ..ElemCo.QMTensors
using ..ElemCo.BasisSets

"""
    Wavefunction

A wavefunction object contains MO coefficients together with the corresponding basis set,
information about frozen orbitals (e.g., frozen core), CC amplitudes, and other information.

All information is stored on disk in a HDF5 file, and is loaded into memory only when needed.

The structure of the HDF5 file is as follows (with `track_order=true`):
```
/EC
  /Molecule1
    <name>
    <geometry>
    /BasisSet1
      <basis set information>
      /State1
        <number of electrons>
        <spin multiplicity>
        <occupation (alpha/beta)>
        <MO coefficients>
        <list of frozen orbitals>
        <CC amplitudes>
        <other information>
      /State2
      ...
    /BasisSet2
    ...
  /Molecule2
    ...
```

The groups can be deleted by calling `delete!(wf; molecule, basis, state)` or 
kept (i.e., deleting everything else) by calling `keep!(wf; molecule, basis, state)`.
"""
@kwdef struct Wavefunction
  filename::String="wf.h5"
end

Wavefunction(EC::AbstractECInfo, filename="wf.h5") = Wavefunction(joinpath(EC.scr, filename))


end #module