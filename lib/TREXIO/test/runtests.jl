"""
Test suite for standalone TREXIO module with new naming convention
"""

using Test
using HDF5

# Load the TREXIO module
push!(LOAD_PATH, joinpath(@__DIR__, "../src"))
using TREXIO

@testset "Standalone TREXIO Tests with New Naming Convention" begin
    
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
            
            # Test opening with new naming convention
            exit_code = TREXIO.trexio_open(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test trexio.file !== nothing
            @test haskey(trexio.file, "trexio")
            
            # Test closing with new naming convention
            exit_code = TREXIO.trexio_close(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test trexio.file === nothing
            
        finally
            isfile(test_file) && rm(test_file)
        end
    end
    
    @testset "Metadata Operations with New Naming Convention" begin
        test_file = "test_metadata.h5"
        
        # Clean up
        isfile(test_file) && rm(test_file)
        
        try
            # Write metadata using new naming convention
            trexio = TREXIO.TrexioFile(test_file, "w")
            exit_code = TREXIO.trexio_write_metadata(trexio, format_version="2.4.0", created_by="Test Suite")
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            # Test has metadata
            @test TREXIO.trexio_has_metadata(trexio) == true
            
            # Read metadata using new naming convention
            metadata, exit_code = TREXIO.trexio_read_metadata(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test haskey(metadata, "format_version")
            @test metadata["format_version"] == "2.4.0"
            @test haskey(metadata, "created_by")
            @test metadata["created_by"] == "Test Suite"
            
            TREXIO.trexio_close(trexio)
            
        finally
            isfile(test_file) && rm(test_file)
        end
    end
    
    @testset "Nucleus Data Operations with New Naming Convention" begin
        test_file = "test_nucleus.h5"
        
        # Clean up
        isfile(test_file) && rm(test_file)
        
        try
            trexio = TREXIO.TrexioFile(test_file, "w")
            
            # Test data
            natoms = 3
            nuclear_charges = [6.0, 1.0, 1.0]
            coordinates = [0.0 1.0 -1.0; 0.0 0.0 0.0; 0.0 0.0 0.0]  # 3×3 matrix (column-major)
            labels = ["C", "H1", "H2"]
            
            # Write nucleus data using new naming convention
            exit_code = TREXIO.trexio_write_nucleus_num(trexio, natoms)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            exit_code = TREXIO.trexio_write_nucleus_charge(trexio, nuclear_charges)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            exit_code = TREXIO.trexio_write_nucleus_coord(trexio, coordinates)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            exit_code = TREXIO.trexio_write_nucleus_label(trexio, labels)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            # Test has functions
            @test TREXIO.trexio_has_nucleus_num(trexio) == true
            @test TREXIO.trexio_has_nucleus_charge(trexio) == true
            @test TREXIO.trexio_has_nucleus_coord(trexio) == true
            @test TREXIO.trexio_has_nucleus_label(trexio) == true
            
            # Read nucleus data using new naming convention
            read_natoms, exit_code = TREXIO.trexio_read_nucleus_num(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test read_natoms == natoms
            
            read_charges, exit_code = TREXIO.trexio_read_nucleus_charge(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test read_charges ≈ nuclear_charges
            
            read_coords, exit_code = TREXIO.trexio_read_nucleus_coord(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test size(read_coords) == (3, natoms)
            @test read_coords ≈ coordinates
            
            read_labels, exit_code = TREXIO.trexio_read_nucleus_label(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test read_labels == labels
            
            TREXIO.trexio_close(trexio)
            
        finally
            isfile(test_file) && rm(test_file)
        end
    end
    
    @testset "Electron Data Operations with New Naming Convention" begin
        test_file = "test_electron.h5"
        
        # Clean up
        isfile(test_file) && rm(test_file)
        
        try
            trexio = TREXIO.TrexioFile(test_file, "w")
            
            # Write electron data using new naming convention
            exit_code = TREXIO.trexio_write_electron_num(trexio, 8)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            exit_code = TREXIO.trexio_write_electron_up_num(trexio, 4)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            exit_code = TREXIO.trexio_write_electron_dn_num(trexio, 4)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            # Test has functions
            @test TREXIO.trexio_has_electron_num(trexio) == true
            @test TREXIO.trexio_has_electron_up_num(trexio) == true
            @test TREXIO.trexio_has_electron_dn_num(trexio) == true
            
            # Read electron data using new naming convention
            read_num, exit_code = TREXIO.trexio_read_electron_num(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test read_num == 8
            
            read_up_num, exit_code = TREXIO.trexio_read_electron_up_num(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test read_up_num == 4
            
            read_dn_num, exit_code = TREXIO.trexio_read_electron_dn_num(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test read_dn_num == 4
            
            TREXIO.trexio_close(trexio)
            
        finally
            isfile(test_file) && rm(test_file)
        end
    end
    
    @testset "MO Data Operations with New Naming Convention" begin
        test_file = "test_mo.h5"
        
        # Clean up
        isfile(test_file) && rm(test_file)
        
        try
            trexio = TREXIO.TrexioFile(test_file, "w")
            
            # Test MO data
            nmo = 5
            nbasis = 10
            mo_coefficients = rand(nbasis, nmo)  # Column-major format
            mo_energies = [-1.0, -0.5, 0.1, 0.2, 0.3]
            mo_occupations = [2.0, 2.0, 1.0, 0.0, 0.0]
            
            # Write MO data using new naming convention
            exit_code = TREXIO.trexio_write_mo_num(trexio, nmo)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            exit_code = TREXIO.trexio_write_mo_coefficient(trexio, mo_coefficients)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            exit_code = TREXIO.trexio_write_mo_energy(trexio, mo_energies)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            exit_code = TREXIO.trexio_write_mo_occupation(trexio, mo_occupations)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            # Test has functions
            @test TREXIO.trexio_has_mo_num(trexio) == true
            @test TREXIO.trexio_has_mo_coefficient(trexio) == true
            @test TREXIO.trexio_has_mo_energy(trexio) == true
            @test TREXIO.trexio_has_mo_occupation(trexio) == true
            
            # Read MO data using new naming convention
            read_nmo, exit_code = TREXIO.trexio_read_mo_num(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test read_nmo == nmo
            
            read_coefficients, exit_code = TREXIO.trexio_read_mo_coefficient(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test size(read_coefficients) == (nbasis, nmo)
            @test read_coefficients ≈ mo_coefficients
            
            read_energies, exit_code = TREXIO.trexio_read_mo_energy(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test read_energies ≈ mo_energies
            
            read_occupations, exit_code = TREXIO.trexio_read_mo_occupation(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test read_occupations ≈ mo_occupations
            
            TREXIO.trexio_close(trexio)
            
        finally
            isfile(test_file) && rm(test_file)
        end
    end
    
    @testset "Basis Data Operations with New Naming Convention" begin
        test_file = "test_basis.h5"
        
        # Clean up
        isfile(test_file) && rm(test_file)
        
        try
            trexio = TREXIO.TrexioFile(test_file, "w")
            
            # Write basis data using new naming convention
            exit_code = TREXIO.trexio_write_basis_shell_num(trexio, 6)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            exit_code = TREXIO.trexio_write_basis_prim_num(trexio, 18)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            # Test has functions
            @test TREXIO.trexio_has_basis_shell_num(trexio) == true
            @test TREXIO.trexio_has_basis_prim_num(trexio) == true
            
            # Read basis data using new naming convention
            read_shell_num, exit_code = TREXIO.trexio_read_basis_shell_num(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test read_shell_num == 6
            
            read_prim_num, exit_code = TREXIO.trexio_read_basis_prim_num(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test read_prim_num == 18
            
            TREXIO.trexio_close(trexio)
            
        finally
            isfile(test_file) && rm(test_file)
        end
    end
    
    @testset "Backward Compatibility Tests" begin
        test_file = "test_backward_compat.h5"
        
        # Clean up
        isfile(test_file) && rm(test_file)
        
        try
            # Test legacy functions still work
            trexio = TREXIO.TrexioFile(test_file, "w")
            
            # Test data
            nuclear_charges = [8.0, 1.0, 1.0]  # Water molecule
            coordinates = [0.0 0.0 0.0; 0.0 1.4 -1.1; 0.0 -1.4 -1.1]  # 3×3 matrix
            labels = ["O1", "H1", "H2"]
            
            # Test legacy write functions
            TREXIO.write_metadata(trexio, format_version="2.4.0", created_by="Legacy Test")
            TREXIO.write_nucleus(trexio, nuclear_charges, coordinates, labels)
            
            # Test MO data
            coefficients = rand(Float64, 5, 3)
            TREXIO.write_mo(trexio, coefficients, orbital_type="molecular")
            
            TREXIO.close_trexio(trexio)
            
            # Test legacy read functions
            trexio_read = TREXIO.TrexioFile(test_file, "r")
            
            metadata = TREXIO.read_metadata(trexio_read)
            @test metadata["format_version"] == "2.4.0"
            @test metadata["created_by"] == "Legacy Test"
            
            charges_read, coords_read, labels_read = TREXIO.read_nucleus(trexio_read)
            @test charges_read ≈ nuclear_charges
            @test coords_read ≈ coordinates
            @test labels_read == labels
            
            mo_data = TREXIO.read_mo(trexio_read)
            @test mo_data["coefficient"] ≈ coefficients
            @test mo_data["type"] == "molecular"
            
            TREXIO.close_trexio(trexio_read)
            
            # Test legacy high-level functions
            data = TREXIO.read_trexio_file(test_file)
            @test haskey(data, "metadata")
            @test haskey(data, "nucleus")
            @test haskey(data, "mo")
            
        finally
            isfile(test_file) && rm(test_file)
        end
    end
    
    @testset "Error Handling with TREXIO Exit Codes" begin
        # Test reading non-existent file
        data, exit_code = TREXIO.trexio_read_file("nonexistent.h5")
        @test exit_code == TREXIO.TREXIO_FILE_ERROR
        
        # Test invalid coordinate dimensions
        test_file = "test_error.h5"
        isfile(test_file) && rm(test_file)
        
        try
            trexio = TREXIO.TrexioFile(test_file, "w")
            
            # Invalid coordinate matrix dimensions (should be 3×n, not 2×n)
            invalid_coords = rand(2, 3)
            exit_code = TREXIO.trexio_write_nucleus_coord(trexio, invalid_coords)
            @test exit_code == TREXIO.TREXIO_INVALID_ARG_2
            
            TREXIO.trexio_close(trexio)
        finally
            isfile(test_file) && rm(test_file)
        end
    end

end