"""
Example demonstrating standalone TREXIO usage independent of ElemCo.jl

This example shows how the TREXIO module can be used independently
for reading and writing quantum chemistry data in TREXIO format.
"""

# Add the standalone TREXIO module to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "../lib/TREXIO/src"))

using TREXIO
using HDF5

println("=== Standalone TREXIO Usage Example ===")

# Example 1: Create a simple molecular system
println("\n1. Creating a water molecule in TREXIO format")

# Water molecule data
nuclear_charges = [8.0, 1.0, 1.0]  # O, H, H
coordinates = [0.0  0.0   0.0;      # Oxygen at origin
               0.0  1.43 -1.11;     # Hydrogen 1
               0.0 -1.43 -1.11]     # Hydrogen 2
labels = ["O1", "H1", "H2"]

# Create TREXIO file
trexio = TREXIO.TrexioFile("water_standalone.h5", "w")
TREXIO.write_metadata(trexio, format_version="2.4.0", created_by="Standalone TREXIO Example")
TREXIO.write_nucleus(trexio, nuclear_charges, coordinates, labels)
TREXIO.close_trexio(trexio)

println("✓ Created water_standalone.h5 with molecular data")

# Example 2: Add molecular orbitals
println("\n2. Adding molecular orbitals")

# Simulate some MO coefficients (5 basis functions, 5 MOs)
mo_coefficients = [
    0.5  0.3  0.0  0.1  0.0;
    0.4  0.4  0.2  0.0  0.1;
    0.3  0.2  0.5  0.2  0.0;
    0.2  0.1  0.3  0.6  0.2;
    0.1  0.0  0.0  0.1  0.8
]

trexio = TREXIO.TrexioFile("water_standalone.h5", "r+")
TREXIO.write_mo(trexio, mo_coefficients, orbital_type="restricted")
TREXIO.close_trexio(trexio)

println("✓ Added molecular orbitals to the file")

# Example 3: Read all data back
println("\n3. Reading data back from TREXIO file")

data = TREXIO.read_trexio_file("water_standalone.h5")
println("Available data sections: ", keys(data))

if haskey(data, "nucleus")
    nucleus_data = data["nucleus"]
    println("Number of atoms: ", length(nucleus_data["charge"]))
    println("Nuclear charges: ", nucleus_data["charge"])
end

if haskey(data, "mo")
    mo_data = data["mo"]
    println("MO matrix size: ", size(mo_data["coefficient"]))
    println("Number of MOs: ", mo_data["num"])
end

if haskey(data, "metadata")
    metadata = data["metadata"]
    println("Format version: ", metadata["format_version"])
    println("Created by: ", metadata["created_by"])
end

# Example 4: High-level API
println("\n4. Using high-level API")

# Create a methane molecule using high-level API
ch4_charges = [6.0, 1.0, 1.0, 1.0, 1.0]  # C, H, H, H, H
ch4_coords = [0.0  0.0  0.0  0.0  0.0;     # x coordinates
              0.0  1.0 -1.0  0.0  0.0;     # y coordinates  
              0.0  0.0  0.0  1.0 -1.0]     # z coordinates
ch4_labels = ["C1", "H1", "H2", "H3", "H4"]

ch4_orbitals = rand(9, 9)  # 9 basis functions for minimal basis CH4

TREXIO.create_trexio_file("methane_standalone.h5", 
                         (ch4_charges, ch4_coords, ch4_labels),
                         nothing,  # no basis data
                         ch4_orbitals,
                         created_by="High-level API example")

println("✓ Created methane_standalone.h5 using high-level API")

# Verify it works
ch4_data = TREXIO.read_trexio_file("methane_standalone.h5")
println("CH4 atoms: ", length(ch4_data["nucleus"]["charge"]))
println("CH4 MOs: ", size(ch4_data["mo"]["coefficient"]))

# Example 5: Demonstrate independence from quantum chemistry packages
println("\n5. Pure data exchange example")

# Simulate receiving data from another quantum chemistry code
external_data = Dict(
    "atoms" => 3,
    "atomic_numbers" => [1, 1, 0],  # H2 + ghost atom
    "positions" => [0.0 0.0 0.0; 0.0 0.0 1.4; 0.0 0.0 2.8],
    "atom_labels" => ["H1", "H2", "X1"],
    "molecular_orbitals" => rand(6, 6)
)

# Convert and store in TREXIO format
h2_file = TREXIO.TrexioFile("h2_exchange.h5", "w")
TREXIO.write_metadata(h2_file, created_by="Data exchange example")
TREXIO.write_nucleus(h2_file, 
                    Float64.(external_data["atomic_numbers"]), 
                    external_data["positions"],
                    external_data["atom_labels"])
TREXIO.write_mo(h2_file, external_data["molecular_orbitals"])
TREXIO.close_trexio(h2_file)

println("✓ Converted external data to TREXIO format")
println("✓ File h2_exchange.h5 can be read by any TREXIO-compatible code")

# Clean up
for file in ["water_standalone.h5", "methane_standalone.h5", "h2_exchange.h5"]
    if isfile(file)
        rm(file)
        println("Removed $file")
    end
end

println("\n=== Standalone TREXIO Example Complete ===")
println("The TREXIO module operates completely independently of any")
println("quantum chemistry package and follows the TREXIO standard.")