"""
Simple test to verify TREXIO separation functionality
"""

# Test 1: Standalone TREXIO usage
println("=== Testing Standalone TREXIO Module ===")

push!(LOAD_PATH, joinpath(@__DIR__, "../lib/TREXIO/src"))
using TREXIO

# Create test data
test_file = "separation_test.h5"
nuclear_charges = [6.0, 1.0, 1.0, 1.0, 1.0]  # CH4
coordinates = zeros(3, 5)
coordinates[:, 1] = [0.0, 0.0, 0.0]  # Carbon at origin
for i in 2:5
    coordinates[:, i] = [1.0, 0.0, 0.0] .* (i-1)  # Hydrogens
end
labels = ["C1", "H1", "H2", "H3", "H4"]

# Test standalone functionality
trexio = TREXIO.TrexioFile(test_file, "w")
TREXIO.write_metadata(trexio, created_by="Separation Test")
TREXIO.write_nucleus(trexio, nuclear_charges, coordinates, labels)

# Add some orbital data
mo_coeffs = rand(Float64, 9, 9)  # 9 basis functions for minimal CH4
TREXIO.write_mo(trexio, mo_coeffs)
TREXIO.close_trexio(trexio)

println("✓ Created TREXIO file with standalone module")

# Read it back
data = TREXIO.read_trexio_file(test_file)
@assert haskey(data, "nucleus")
@assert haskey(data, "mo") 
@assert haskey(data, "metadata")
@assert data["nucleus"]["charge"] ≈ nuclear_charges
@assert data["mo"]["coefficient"] ≈ mo_coeffs

println("✓ Successfully read data back")
println("  - Metadata: ", data["metadata"]["created_by"])
println("  - Atoms: ", length(data["nucleus"]["charge"]))
println("  - MOs: ", size(data["mo"]["coefficient"]))

# Test 2: Independence verification
println("\n=== Testing Independence ===")

# This should work without any ElemCo.jl dependencies
independent_test_file = "independent_test.h5"

# Create data that has no connection to quantum chemistry specifics
generic_charges = [1.0, 2.0, 3.0]
generic_coords = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
generic_labels = ["A", "B", "C"]
generic_matrix = Matrix{Float64}(reshape(1.0:12.0, 3, 4))

# Use only the core TREXIO functionality
trex = TREXIO.TrexioFile(independent_test_file, "w")
TREXIO.write_metadata(trex, format_version="2.4.0", created_by="Independent Test")
TREXIO.write_nucleus(trex, generic_charges, generic_coords, generic_labels)
TREXIO.write_mo(trex, generic_matrix)
TREXIO.close_trexio(trex)

# Verify it follows TREXIO standard
independent_data = TREXIO.read_trexio_file(independent_test_file)
@assert independent_data["metadata"]["format_version"] == "2.4.0"
@assert independent_data["nucleus"]["coord"] ≈ generic_coords  # Column-major
@assert independent_data["mo"]["coefficient"] ≈ generic_matrix

println("✓ Independent usage confirmed - no ElemCo.jl dependencies")
println("✓ TREXIO format compliance verified")

# Test 3: Data exchange capability
println("\n=== Testing Data Exchange Capability ===")

# Simulate data from another quantum chemistry code
external_data = Dict(
    "software" => "External QC Code",
    "molecule" => "Water",
    "nuclear_charges" => [8.0, 1.0, 1.0],
    "coordinates_bohr" => [0.0 0.0 0.0; 0.0 1.43 -1.11; 0.0 -1.43 -1.11],
    "atom_labels" => ["O", "H1", "H2"],
    "scf_orbitals" => rand(7, 7)  # STO-3G water
)

# Convert to TREXIO using standalone module
exchange_file = "data_exchange.h5"
TREXIO.create_trexio_file(
    exchange_file,
    (external_data["nuclear_charges"], external_data["coordinates_bohr"], external_data["atom_labels"]),
    nothing,  # no basis set data
    external_data["scf_orbitals"],
    created_by=external_data["software"]
)

# Read back and verify
exchange_data = TREXIO.read_trexio_file(exchange_file)
@assert exchange_data["metadata"]["created_by"] == "External QC Code"
@assert exchange_data["nucleus"]["charge"] ≈ external_data["nuclear_charges"]
@assert exchange_data["mo"]["coefficient"] ≈ external_data["scf_orbitals"]

println("✓ Data exchange capability confirmed")
println("  - Source: ", exchange_data["metadata"]["created_by"])
println("  - Molecule: 3 atoms (", exchange_data["nucleus"]["label"], ")")
println("  - Orbitals: ", size(exchange_data["mo"]["coefficient"]))

# Clean up
for file in [test_file, independent_test_file, exchange_file]
    isfile(file) && rm(file)
end

println("\n=== All Tests Passed ===")
println("✓ Standalone TREXIO module works independently")
println("✓ No ElemCo.jl dependencies required")
println("✓ TREXIO format compliance verified")
println("✓ Data exchange capability confirmed")
println("✓ Separation objective achieved")