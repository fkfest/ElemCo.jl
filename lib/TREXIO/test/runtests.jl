"""
Test suite for standalone TREXIO module with generated function API
"""

using Test
using HDF5

# Load the TREXIO module
push!(LOAD_PATH, joinpath(@__DIR__, "../src"))
using TREXIO

@testset "TREXIO Tests" begin
    
    @testset "TrexioFile Basic Operations" begin
        test_file = "test_basic.h5"
        
        # Clean up
        isfile(test_file) && rm(test_file)
        
        try
            # Test file creation with trexio_open (returns TrexioFile directly)
            trexio = TREXIO.trexio_open(test_file, "w")
            @test trexio.filename == test_file
            @test trexio.mode == "w"
            @test trexio.file !== nothing
            @test isopen(trexio.file)
            
            # Test closing
            exit_code = TREXIO.trexio_close(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
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
            trexio = TREXIO.trexio_open(test_file, "w")
            
            exit_code = TREXIO.trexio_write_metadata_package_version(trexio, "2.4.0")
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            exit_code = TREXIO.trexio_write_metadata_description(trexio, "Test file")
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            # Test has metadata
            @test TREXIO.trexio_has_metadata_package_version(trexio)
            @test TREXIO.trexio_has_metadata_description(trexio)
            
            # Read metadata
            version, exit_code = TREXIO.trexio_read_metadata_package_version(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test version == "2.4.0"
            
            description, exit_code = TREXIO.trexio_read_metadata_description(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test description == "Test file"
            
            TREXIO.trexio_close(trexio)
            
        finally
            isfile(test_file) && rm(test_file)
        end
    end
    
    @testset "Nucleus Data Operations" begin
        test_file = "test_nucleus.h5"
        
        # Clean up
        isfile(test_file) && rm(test_file)
        
        try
            trexio = TREXIO.trexio_open(test_file, "w")
            
            # Test data
            natoms = 3
            nuclear_charges = [6.0, 1.0, 1.0]
            coordinates = [0.0 1.0 -1.0; 0.0 0.0 0.0; 0.0 0.0 0.0]  # 3×3 matrix (column-major)
            labels = ["C", "H1", "H2"]
            
            # Write nucleus data (num must be written first for size validation)
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
            
            # Read nucleus data
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
    
    @testset "Electron Data Operations" begin
        test_file = "test_electron.h5"
        
        # Clean up
        isfile(test_file) && rm(test_file)
        
        try
            trexio = TREXIO.trexio_open(test_file, "w")
            
            # Write electron data
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
            
            # Read electron data
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
    
    @testset "MO Data Operations" begin
        test_file = "test_mo.h5"
        
        # Clean up
        isfile(test_file) && rm(test_file)
        
        try
            trexio = TREXIO.trexio_open(test_file, "w")
            
            # Test MO data
            nmo = 5
            nao = 10
            mo_coefficients = rand(nao, nmo)  # Column-major format: (ao.num, mo.num)
            mo_energies = [-1.0, -0.5, 0.1, 0.2, 0.3]
            mo_occupations = [2.0, 2.0, 1.0, 0.0, 0.0]
            
            # Write dimensions first (required for array size validation)
            exit_code = TREXIO.trexio_write_mo_num(trexio, nmo)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            exit_code = TREXIO.trexio_write_ao_num(trexio, nao)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            # Write MO data
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
            
            # Read MO data
            read_nmo, exit_code = TREXIO.trexio_read_mo_num(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test read_nmo == nmo
            
            read_coefficients, exit_code = TREXIO.trexio_read_mo_coefficient(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test size(read_coefficients) == (nao, nmo)
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
    
    @testset "Basis Data Operations" begin
        test_file = "test_basis.h5"
        
        # Clean up
        isfile(test_file) && rm(test_file)
        
        try
            trexio = TREXIO.trexio_open(test_file, "w")
            
            # Write basis data
            exit_code = TREXIO.trexio_write_basis_shell_num(trexio, 6)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            exit_code = TREXIO.trexio_write_basis_prim_num(trexio, 18)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            exit_code = TREXIO.trexio_write_basis_type(trexio, "Gaussian")
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            # Test has functions
            @test TREXIO.trexio_has_basis_shell_num(trexio) == true
            @test TREXIO.trexio_has_basis_prim_num(trexio) == true
            @test TREXIO.trexio_has_basis_type(trexio) == true
            
            # Read basis data
            read_shell_num, exit_code = TREXIO.trexio_read_basis_shell_num(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test read_shell_num == 6
            
            read_prim_num, exit_code = TREXIO.trexio_read_basis_prim_num(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test read_prim_num == 18
            
            read_type, exit_code = TREXIO.trexio_read_basis_type(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test read_type == "Gaussian"
            
            TREXIO.trexio_close(trexio)
            
        finally
            isfile(test_file) && rm(test_file)
        end
    end
    
    @testset "Sparse Array Operations" begin
        test_file = "test_sparse.h5"
        
        # Clean up
        isfile(test_file) && rm(test_file)
        
        try
            trexio = TREXIO.trexio_open(test_file, "w")
            
            # Write dimensions first
            nmo = 4
            exit_code = TREXIO.trexio_write_mo_num(trexio, nmo)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            # Test sparse MO 2e integrals (4 indices)
            # Indices: 4×n_elements, Values: n_elements
            n_elements = 5
            indices = Int32[1 1 2 2 3; 1 2 1 2 3; 1 1 2 2 3; 1 2 1 2 3]  # 4×5 matrix
            values = Float64[0.5, 0.3, 0.2, 0.4, 0.1]
            
            exit_code = TREXIO.trexio_write_mo_2e_int_eri(trexio, (indices, values))
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            @test TREXIO.trexio_has_mo_2e_int_eri(trexio) == true
            
            # Read sparse data
            (read_indices, read_values), exit_code = TREXIO.trexio_read_mo_2e_int_eri(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test read_indices == indices
            @test read_values ≈ values
            
            TREXIO.trexio_close(trexio)
            
        finally
            isfile(test_file) && rm(test_file)
        end
    end
    
    @testset "State Data Operations" begin
        test_file = "test_state.h5"
        
        # Clean up
        isfile(test_file) && rm(test_file)
        
        try
            trexio = TREXIO.trexio_open(test_file, "w")
            
            # Write state data
            exit_code = TREXIO.trexio_write_state_num(trexio, 3)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            exit_code = TREXIO.trexio_write_state_id(trexio, 0)  # Ground state
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            exit_code = TREXIO.trexio_write_state_energy(trexio, -75.5)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            exit_code = TREXIO.trexio_write_state_current_label(trexio, "Ground")
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            # Test has functions
            @test TREXIO.trexio_has_state_num(trexio) == true
            @test TREXIO.trexio_has_state_id(trexio) == true
            @test TREXIO.trexio_has_state_energy(trexio) == true
            @test TREXIO.trexio_has_state_current_label(trexio) == true
            
            # Read state data
            read_num, exit_code = TREXIO.trexio_read_state_num(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test read_num == 3
            
            read_id, exit_code = TREXIO.trexio_read_state_id(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test read_id == 0
            
            read_energy, exit_code = TREXIO.trexio_read_state_energy(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test read_energy ≈ -75.5
            
            read_label, exit_code = TREXIO.trexio_read_state_current_label(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test read_label == "Ground"
            
            TREXIO.trexio_close(trexio)
            
        finally
            isfile(test_file) && rm(test_file)
        end
    end
    
    @testset "AO One-Electron Integrals" begin
        test_file = "test_ao_1e.h5"
        
        # Clean up
        isfile(test_file) && rm(test_file)
        
        try
            trexio = TREXIO.trexio_open(test_file, "w")
            
            # Write dimensions first
            nao = 5
            exit_code = TREXIO.trexio_write_ao_num(trexio, nao)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            # Create test matrices
            overlap = zeros(nao, nao)
            for i in 1:nao
                overlap[i, i] = 1.0
            end
            kinetic = rand(nao, nao)
            kinetic = 0.5 * (kinetic + kinetic')  # Make symmetric
            
            # Write integrals
            exit_code = TREXIO.trexio_write_ao_1e_int_overlap(trexio, overlap)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            exit_code = TREXIO.trexio_write_ao_1e_int_kinetic(trexio, kinetic)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            
            # Test has functions
            @test TREXIO.trexio_has_ao_1e_int_overlap(trexio) == true
            @test TREXIO.trexio_has_ao_1e_int_kinetic(trexio) == true
            
            # Read integrals
            read_overlap, exit_code = TREXIO.trexio_read_ao_1e_int_overlap(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test read_overlap ≈ overlap
            
            read_kinetic, exit_code = TREXIO.trexio_read_ao_1e_int_kinetic(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test read_kinetic ≈ kinetic
            
            TREXIO.trexio_close(trexio)
            
        finally
            isfile(test_file) && rm(test_file)
        end
    end
    
    @testset "Error Handling" begin
        # Test reading non-existent attributes
        test_file = "test_error.h5"
        isfile(test_file) && rm(test_file)
        
        try
            trexio = TREXIO.trexio_open(test_file, "w")
            
            # Try to read non-existent data
            _, exit_code = TREXIO.trexio_read_nucleus_num(trexio)
            @test exit_code == TREXIO.TREXIO_HAS_NOT
            
            _, exit_code = TREXIO.trexio_read_mo_coefficient(trexio)
            @test exit_code == TREXIO.TREXIO_HAS_NOT
            
            # Test invalid array dimensions
            # Write nucleus_num first
            TREXIO.trexio_write_nucleus_num(trexio, 3)
            
            # Try to write coordinates with wrong first dimension (should be 3)
            invalid_coords = rand(2, 3)  # 2×3 instead of 3×3
            exit_code = TREXIO.trexio_write_nucleus_coord(trexio, invalid_coords)
            @test exit_code == TREXIO.TREXIO_INVALID_ARG_2
            
            # Test scalar type validation
            exit_code = TREXIO.trexio_write_electron_num(trexio, 8.5)  # Float instead of Int
            @test exit_code == TREXIO.TREXIO_INVALID_ARG_2
            
            TREXIO.trexio_close(trexio)
        finally
            isfile(test_file) && rm(test_file)
        end
    end
    
    @testset "File Reopen and Update" begin
        test_file = "test_reopen.h5"
        
        # Clean up
        isfile(test_file) && rm(test_file)
        
        try
            # Write initial data
            trexio = TREXIO.trexio_open(test_file, "w")
            TREXIO.trexio_write_nucleus_num(trexio, 2)
            TREXIO.trexio_write_nucleus_charge(trexio, [8.0, 1.0])
            TREXIO.trexio_close(trexio)
            
            # Reopen and read
            trexio = TREXIO.trexio_open(test_file, "r")
            natoms, exit_code = TREXIO.trexio_read_nucleus_num(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test natoms == 2
            
            charges, exit_code = TREXIO.trexio_read_nucleus_charge(trexio)
            @test exit_code == TREXIO.TREXIO_SUCCESS
            @test charges ≈ [8.0, 1.0]
            TREXIO.trexio_close(trexio)
            
            # Reopen in update mode and add more data
            trexio = TREXIO.trexio_open(test_file, "u")
            TREXIO.trexio_write_electron_num(trexio, 9)
            TREXIO.trexio_close(trexio)
            
            # Verify all data is present
            trexio = TREXIO.trexio_open(test_file, "r")
            natoms, _ = TREXIO.trexio_read_nucleus_num(trexio)
            @test natoms == 2
            nelec, _ = TREXIO.trexio_read_electron_num(trexio)
            @test nelec == 9
            TREXIO.trexio_close(trexio)
            
        finally
            isfile(test_file) && rm(test_file)
        end
    end
    
    @testset "Utility Functions" begin
        test_file = "test_utilities.h5"
        
        # Clean up
        isfile(test_file) && rm(test_file)
        
        try
            trexio = TREXIO.trexio_open(test_file, "w")
            
            # Write some data
            TREXIO.trexio_write_nucleus_num(trexio, 3)
            
            # Test check_read_status function
            num, status = TREXIO.trexio_read_nucleus_num(trexio)
            @test_nowarn TREXIO.trexio_check_read_status(status, "nucleus_num")
            
            # Test check_write_status function (should not throw for success)
            status = TREXIO.trexio_write_electron_num(trexio, 10)
            @test_nowarn TREXIO.trexio_check_write_status(status, "electron_num")
            
            TREXIO.trexio_close(trexio)
            
        finally
            isfile(test_file) && rm(test_file)
        end
    end

end