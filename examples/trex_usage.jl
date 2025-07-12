"""
TREX Interface Usage Examples

This file demonstrates how to use the TREX format interface in ElemCo.jl
for storing and retrieving quantum chemistry data.
"""

using ElemCo

# Set the output to suppress verbose output for examples
@print_input

# Example 1: Basic TREX export with molecular geometry
println("=== Example 1: Export molecular geometry to TREX ===")

# Define a water molecule
geometry = "bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"

basis = "sto-3g"

# Initialize the calculation
@ECinit

# Export just the molecular structure to TREX format
@write_trex "water_geometry.h5" include_orbitals=false include_amplitudes=false

println("Water geometry exported to water_geometry.h5")


# Example 2: Export with orbitals after HF calculation
println("\n=== Example 2: Export with HF orbitals ===")

try
    # Perform a simple HF calculation
    @dfhf
    
    # Export geometry and orbitals to TREX format
    @write_trex "water_hf.h5" include_orbitals=true include_amplitudes=false
    
    println("Water HF calculation exported to water_hf.h5")
catch e
    println("HF calculation skipped: $e")
end


# Example 3: Reading TREX data
println("\n=== Example 3: Reading TREX data ===")

try
    # Read the TREX file
    data = @read_trex "water_geometry.h5"
    
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
    println("Reading TREX data failed: $e")
end


# Example 4: Using low-level TREX interface
println("\n=== Example 4: Low-level TREX interface ===")

try
    using ElemCo.TrexInterface
    
    # Create a TREX file manually
    trex = TrexFile("manual_trex.h5", "w")
    
    # Write molecular data
    if !isnothing(EC.system)
        write_trex_molecule(trex, EC.system)
        println("Molecular data written to manual_trex.h5")
    end
    
    # Write some test orbital data
    test_orbitals = rand(Float64, 7, 7)  # STO-3G for water has 7 basis functions
    write_trex_orbitals(trex, test_orbitals)
    println("Test orbital data written")
    
    # Close the file
    close_trex(trex)
    
    # Read back the data
    trex_read = TrexFile("manual_trex.h5", "r")
    
    molecule_read = read_trex_molecule(trex_read)
    orbitals_read = read_trex_orbitals(trex_read)
    
    println("Read back: $(length(molecule_read)) atoms, $(size(orbitals_read)) orbital matrix")
    
    close_trex(trex_read)
    
catch e
    println("Low-level interface failed: $e")
end


# Example 5: TREX format for data exchange
println("\n=== Example 5: Data exchange workflow ===")

println("""
Typical TREX workflow for data exchange:

1. Export calculation results:
   @dfhf
   @cc ccsd
   @write_trex "results.h5" include_amplitudes=true

2. Share the TREX file with collaborators

3. Import data in another calculation:
   data = @read_trex "results.h5"
   # Use data["molecule"], data["orbitals"], data["amplitudes"] as needed

4. The TREX format ensures compatibility across different
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

println("\nTREX interface examples completed!")