"""
Test suite for standalone TREXIO module
"""

using Test
using HDF5

# Load the TREXIO module
push!(LOAD_PATH, joinpath(@__DIR__, "../src"))
using TREXIO

@testset "Standalone TREXIO Tests" begin
    
    @testset "TrexioFile Basic Operations" begin
        test_file = "test_basic.h5"
        
        # Clean up
        isfile(test_file) && rm(test_file)
        
        try
            # Test file creation and opening
            trexio = TREXIO.TrexioFile(test_file, "w")
            @test trexio.filename == test_file
            @test trexio.mode == "w"
            @test trexio.file === nothing
            
            # Test opening
            file = TREXIO.open_trexio(trexio)
            @test file !== nothing
            @test haskey(file, "trexio")
            
            # Test closing
            TREXIO.close_trexio(trexio)
            @test trexio.file === nothing
            
        finally
            isfile(test_file) && rm(test_file)
        end
    end
    
    @testset "Metadata Operations" begin
        test_file = "test_metadata.h5"
        
        # Clean up
        isfile(test_file) && rm(test_file)
        
        try
            # Write metadata
            trexio = TREXIO.TrexioFile(test_file, "w")
            TREXIO.write_metadata(trexio, format_version="2.4.0", created_by="Test Suite")
            TREXIO.close_trexio(trexio)
            
            @test isfile(test_file)
            
            # Read metadata
            trexio_read = TREXIO.TrexioFile(test_file, "r")
            metadata = TREXIO.read_metadata(trexio_read)
            TREXIO.close_trexio(trexio_read)
            
            @test metadata["format_version"] == "2.4.0"
            @test metadata["created_by"] == "Test Suite"
            @test haskey(metadata, "created_date")
            
        finally
            isfile(test_file) && rm(test_file)
        end
    end
    
    @testset "Nucleus Data Operations" begin
        test_file = "test_nucleus.h5"
        
        # Clean up
        isfile(test_file) && rm(test_file)
        
        try
            # Test data (Water molecule)
            nuclear_charges = [8.0, 1.0, 1.0]  # O, H, H
            coordinates = [0.0 0.0 0.0; 0.0 1.4 -1.1; 0.0 -1.4 -1.1]  # 3×3 matrix (column-major)
            labels = ["O1", "H1", "H2"]
            
            # Write nucleus data
            trexio = TREXIO.TrexioFile(test_file, "w")
            TREXIO.write_nucleus(trexio, nuclear_charges, coordinates, labels)
            TREXIO.close_trexio(trexio)
            
            @test isfile(test_file)
            
            # Read nucleus data back
            trexio_read = TREXIO.TrexioFile(test_file, "r")
            charges_read, coords_read, labels_read = TREXIO.read_nucleus(trexio_read)
            TREXIO.close_trexio(trexio_read)
            
            @test charges_read ≈ nuclear_charges
            @test coords_read ≈ coordinates
            @test labels_read == labels
            
        finally
            isfile(test_file) && rm(test_file)
        end
    end
    
    @testset "Basis Set Data Operations" begin
        test_file = "test_basis.h5"
        
        # Clean up
        isfile(test_file) && rm(test_file)
        
        try
            # Test basis set data (minimal example)
            shell_num = 2
            shell_nucleus_index = [1, 1]  # Both shells on first atom
            shell_ang_mom = [0, 0]  # Both s shells
            shell_factor = [1.0, 1.0]
            shell_range = [2, 1]  # First shell has 2 primitives, second has 1
            exponent = [13.0, 1.96, 0.444]  # STO-3G like exponents
            coefficient = [0.15, 0.85, 1.0]  # Coefficients
            
            # Write basis data
            trexio = TREXIO.TrexioFile(test_file, "w")
            TREXIO.write_basis(trexio, shell_num, shell_nucleus_index, shell_ang_mom,
                              shell_factor, shell_range, exponent, coefficient)
            TREXIO.close_trexio(trexio)
            
            @test isfile(test_file)
            
            # Read basis data back
            trexio_read = TREXIO.TrexioFile(test_file, "r")
            basis_data = TREXIO.read_basis(trexio_read)
            TREXIO.close_trexio(trexio_read)
            
            @test basis_data["shell_num"] == shell_num
            @test basis_data["prim_num"] == length(exponent)
            @test basis_data["shell_nucleus_index"] == shell_nucleus_index
            @test basis_data["shell_ang_mom"] == shell_ang_mom
            @test basis_data["shell_factor"] ≈ shell_factor
            @test basis_data["shell_range"] == shell_range
            @test basis_data["exponent"] ≈ exponent
            @test basis_data["coefficient"] ≈ coefficient
            
        finally
            isfile(test_file) && rm(test_file)
        end
    end
    
    @testset "MO Data Operations" begin
        test_file = "test_mo.h5"
        
        # Clean up
        isfile(test_file) && rm(test_file)
        
        try
            # Test MO data
            coefficients = rand(Float64, 5, 3)  # 5 basis functions, 3 MOs
            orbital_type = "molecular"
            
            # Write MO data
            trexio = TREXIO.TrexioFile(test_file, "w")
            TREXIO.write_mo(trexio, coefficients, orbital_type=orbital_type)
            TREXIO.close_trexio(trexio)
            
            @test isfile(test_file)
            
            # Read MO data back
            trexio_read = TREXIO.TrexioFile(test_file, "r")
            mo_data = TREXIO.read_mo(trexio_read)
            TREXIO.close_trexio(trexio_read)
            
            @test mo_data["coefficient"] ≈ coefficients
            @test mo_data["num"] == size(coefficients, 2)
            @test mo_data["type"] == orbital_type
            @test mo_data["basis_size"] == size(coefficients, 1)
            
        finally
            isfile(test_file) && rm(test_file)
        end
    end
    
    @testset "High-level File Operations" begin
        test_file = "test_complete.h5"
        
        # Clean up
        isfile(test_file) && rm(test_file)
        
        try
            # Create complete test data
            nuclear_charges = [6.0, 1.0, 1.0, 1.0, 1.0]  # CH4
            coordinates = zeros(3, 5)  # Simple geometry
            coordinates[:, 1] = [0.0, 0.0, 0.0]  # Carbon at origin
            for i in 2:5
                coordinates[:, i] = [1.0, 0.0, 0.0] .* (i-1)  # Hydrogens along x-axis
            end
            labels = ["C1", "H1", "H2", "H3", "H4"]
            
            mo_coefficients = rand(Float64, 10, 8)  # 10 basis functions, 8 MOs
            
            nucleus_data = (nuclear_charges, coordinates, labels)
            
            # Create file with high-level function
            TREXIO.create_trexio_file(test_file, nucleus_data, nothing, mo_coefficients,
                                     created_by="High-level test")
            
            @test isfile(test_file)
            
            # Read back with high-level function
            data = TREXIO.read_trexio_file(test_file)
            
            @test haskey(data, "metadata")
            @test haskey(data, "nucleus") 
            @test haskey(data, "mo")
            @test data["metadata"]["created_by"] == "High-level test"
            @test data["nucleus"]["charge"] ≈ nuclear_charges
            @test data["nucleus"]["coord"] ≈ coordinates
            @test data["nucleus"]["label"] == labels
            @test data["mo"]["coefficient"] ≈ mo_coefficients
            
        finally
            isfile(test_file) && rm(test_file)
        end
    end
    
    @testset "Error Handling" begin
        # Test reading non-existent file
        @test_throws Exception TREXIO.read_trexio_file("nonexistent.h5")
        
        # Test invalid coordinate dimensions
        test_file = "test_error.h5"
        isfile(test_file) && rm(test_file)
        
        try
            trexio = TREXIO.TrexioFile(test_file, "w")
            
            # Invalid coordinate matrix dimensions
            @test_throws Exception TREXIO.write_nucleus(trexio, [1.0], 
                                                       rand(2, 2), ["H"])  # Should be 3×1
            
            TREXIO.close_trexio(trexio)
        finally
            isfile(test_file) && rm(test_file)
        end
    end
end