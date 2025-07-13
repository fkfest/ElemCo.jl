"""
TREXIO Interface Usage Examples

This file demonstrates how to use the TREXIO format interface in ElemCo.jl
for storing and retrieving quantum chemistry data.
"""

using ElemCo

# Set the output to suppress verbose output for examples
@print_input

# Example 1: Basic TREXIO export with molecular geometry
println("=== Example 1: Export molecular geometry to TREXIO ===")

# Define a water molecule
geometry = "bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"

basis = "sto-3g"

# Initialize the calculation
@ECinit

# Export just the molecular structure to TREXIO format
@write_trexioio "water_geometry.h5" include_orbitals=false include_amplitudes=false

println("Water geometry exported to water_geometry.h5")


# Example 2: Export with orbitals after HF calculation
println("\n=== Example 2: Export with HF orbitals ===")

try
    # Perform a simple HF calculation
    @dfhf
    
    # Export geometry and orbitals to TREXIO format
    @write_trexioio "water_hf.h5" include_orbitals=true include_amplitudes=false
    
    println("Water HF calculation exported to water_hf.h5")
catch e
    println("HF calculation skipped: $e")
end


# Example 3: Reading TREXIO data
println("\n=== Example 3: Reading TREXIO data ===")

try
    # Read the TREXIO file
    data = @read_trexioio "water_geometry.h5"
    
    println("Available data sections: ", keys(data))
    
    if haskey(data, "molecule")
        mol = data["molecule"]
        println("Number of atoms: ", length(mol))
        for (i, atom) in enumerate(mol)
            println("Atom $i: $(atom.label) at position $(atom.position)")
        end
    end
    
    if haskey(data, "orbitals")
        orbs = data["orbitals"]
        println("Orbital matrix size: ", size(orbs))
    end
    
catch e
    println("Reading TREXIO data failed: $e")
end


# Example 4: Using low-level TREXIO interface
println("\n=== Example 4: Low-level TREXIO interface ===")

try
    using ElemCo.TrexioInterface
    
    # Create a TREXIO file manually
    trex = TrexioFile("manual_trex.h5", "w")
    
    # Write molecular data
    if !isnothing(EC.system)
        write_trexio_molecule(trex, EC.system)
        println("Molecular data written to manual_trex.h5")
    end
    
    # Write some test orbital data
    test_orbitals = rand(Float64, 7, 7)  # STO-3G for water has 7 basis functions
    write_trexio_orbitals(trex, test_orbitals)
    println("Test orbital data written")
    
    # Close the file
    ElemCo.TrexioInterface.close_trexio(trex)
    
    # Read back the data
    trex_read = TrexioFile("manual_trex.h5", "r")
    
    molecule_read = read_trexio_molecule(trex_read)
    orbitals_read = read_trexio_orbitals(trex_read)
    
    println("Read back: $(length(molecule_read)) atoms, $(size(orbitals_read)) orbital matrix")
    
    ElemCo.TrexioInterface.close_trexio(trex_read)
    
catch e
    println("Low-level interface failed: $e")
end


# Example 5: TREXIO format for data exchange
println("\n=== Example 5: Data exchange workflow ===")

println("""
Typical TREXIO workflow for data exchange:

1. Export calculation results:
   @dfhf
   @cc ccsd
   @write_trexioio "results.h5" include_amplitudes=true

2. Share the TREXIO file with collaborators

3. Import data in another calculation:
   data = @read_trexioio "results.h5"
   # Use data["molecule"], data["orbitals"], data["amplitudes"] as needed

4. The TREXIO format ensures compatibility across different
   quantum chemistry codes that support the standard.
""")


# Clean up example files
println("\n=== Cleaning up example files ===")
for file in ["water_geometry.h5", "water_hf.h5", "manual_trex.h5"]
    if isfile(file)
        rm(file)
        println("Removed $file")
    end
end

println("\nTREXIO interface examples completed!")