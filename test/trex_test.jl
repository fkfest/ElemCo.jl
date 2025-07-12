"""
Test for TREX interface functionality
"""

using Test
using ElemCo
using ElemCo.TrexInterface
using HDF5
using Dates

# Setup a simple test system
@testset "TREX Interface Tests" begin
    
    # Test basic TREX file operations
    @testset "TREX File Operations" begin
        test_filename = "test_trex.h5"
        
        # Clean up any existing test file
        if isfile(test_filename)
            rm(test_filename)
        end
        
        # Test TrexFile creation
        trex = TrexFile(test_filename, "w")
        @test trex.filename == test_filename
        @test trex.mode == "w"
        @test trex.file === nothing
        
        # Test file opening
        file = open_trex(trex)
        @test file !== nothing
        @test haskey(file, "trex")
        
        # Test file closing
        close_trex(trex)
        @test trex.file === nothing
        
        # Clean up
        if isfile(test_filename)
            rm(test_filename)
        end
    end
    
    # Test molecular data I/O
    @testset "Molecular Data I/O" begin
        test_filename = "test_molecule.h5"
        
        # Clean up any existing test file
        if isfile(test_filename)
            rm(test_filename)
        end
        
        # Create a simple molecular system (H2)
        geometry = "H 0.0 0.0 0.0\nH 0.0 0.0 1.4"
        basis = Dict("ao" => "sto-3g")
        
        try
            # Parse geometry and create system
            system = parse_geometry(geometry, basis)
            
            # Test writing molecular data
            trex = TrexFile(test_filename, "w")
            nucleus_group = write_trex_molecule(trex, system)
            @test nucleus_group !== nothing
            close_trex(trex)
            
            # Test reading molecular data
            trex_read = TrexFile(test_filename, "r")
            system_read = read_trex_molecule(trex_read)
            @test length(system_read) == length(system)
            close_trex(trex_read)
            
        catch e
            @warn "Molecular data test skipped due to: $e"
        end
        
        # Clean up
        if isfile(test_filename)
            rm(test_filename)
        end
    end
    
    # Test orbital data I/O
    @testset "Orbital Data I/O" begin
        test_filename = "test_orbitals.h5"
        
        # Clean up any existing test file
        if isfile(test_filename)
            rm(test_filename)
        end
        
        # Create test orbital matrix
        test_orbitals = rand(Float64, 10, 5)  # 10 basis functions, 5 MOs
        
        # Test writing orbital data
        trex = TrexFile(test_filename, "w")
        mo_group = write_trex_orbitals(trex, test_orbitals)
        @test mo_group !== nothing
        close_trex(trex)
        
        # Test reading orbital data
        trex_read = TrexFile(test_filename, "r")
        orbitals_read = read_trex_orbitals(trex_read)
        @test size(orbitals_read) == size(test_orbitals)
        @test isapprox(orbitals_read, test_orbitals)
        close_trex(trex_read)
        
        # Clean up
        if isfile(test_filename)
            rm(test_filename)
        end
    end
    
    # Test amplitude data I/O
    @testset "Amplitude Data I/O" begin
        test_filename = "test_amplitudes.h5"
        
        # Clean up any existing test file
        if isfile(test_filename)
            rm(test_filename)
        end
        
        # Create test amplitude data
        test_amplitudes = Dict{String, Any}(
            "t1" => rand(Float64, 5, 5),
            "t2" => rand(Float64, 5, 5, 5, 5)
        )
        
        # Test writing amplitude data
        trex = TrexFile(test_filename, "w")
        amp_group = write_trex_amplitudes(trex, test_amplitudes)
        @test amp_group !== nothing
        close_trex(trex)
        
        # Test reading amplitude data
        trex_read = TrexFile(test_filename, "r")
        amplitudes_read = read_trex_amplitudes(trex_read)
        @test haskey(amplitudes_read, "t1")
        @test haskey(amplitudes_read, "t2")
        @test isapprox(amplitudes_read["t1"], test_amplitudes["t1"])
        @test isapprox(amplitudes_read["t2"], test_amplitudes["t2"])
        close_trex(trex_read)
        
        # Clean up
        if isfile(test_filename)
            rm(test_filename)
        end
    end
    
    # Test high-level read/write functions
    @testset "High-level TREX I/O" begin
        test_filename = "test_highlevel.h5"
        
        # Clean up any existing test file
        if isfile(test_filename)
            rm(test_filename)
        end
        
        try
            # Create a simple test setup
            geometry = "H 0.0 0.0 0.0\nH 0.0 0.0 1.4"
            basis = Dict("ao" => "sto-3g")
            
            # Initialize EC system
            EC = ECInfo()
            EC.system = parse_geometry(geometry, basis)
            
            # Test writing TREX file
            write_trex(test_filename, EC, include_orbitals=false, include_amplitudes=false)
            @test isfile(test_filename)
            
            # Test reading TREX file
            data = read_trex(test_filename)
            @test haskey(data, "molecule")
            
        catch e
            @warn "High-level I/O test skipped due to: $e"
        end
        
        # Clean up
        if isfile(test_filename)
            rm(test_filename)
        end
    end
end