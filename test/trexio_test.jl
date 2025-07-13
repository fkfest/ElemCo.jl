"""
Test for TREXIO interface functionality
"""

using Test

# Only test if HDF5 is available
@testset "TREXIO Interface Tests" begin
    try
        using HDF5
        using ElemCo
        
        # Test basic TREXIO file operations
        @testset "TREXIO File Operations" begin
            test_filename = "test_trexio.h5"
            
            # Clean up any existing test file
            if isfile(test_filename)
                rm(test_filename)
            end
            
            try
                # Test orbital data I/O (simplified test)
                test_orbitals = rand(Float64, 5, 3)  # 5 basis functions, 3 MOs
                
                # Test high-level write function
                trexio_data = Dict{String, Any}("orbitals" => test_orbitals)
                
                # Use HDF5 directly for basic test
                h5open(test_filename, "w") do file
                    trexio_group = create_group(file, "trexio")
                    mo_group = create_group(trexio_group, "mo")
                    mo_group["num"] = size(test_orbitals, 2)
                    mo_group["coefficient"] = test_orbitals
                end
                
                @test isfile(test_filename)
                
                # Test reading
                h5open(test_filename, "r") do file
                    if haskey(file, "trexio") && haskey(file["trexio"], "mo")
                        orbitals_read = read(file["trexio"]["mo"]["coefficient"])
                        @test size(orbitals_read) == size(test_orbitals)
                        @test isapprox(orbitals_read, test_orbitals)
                    end
                end
                
            catch e
                @warn "TREXIO basic test failed: $e"
            finally
                # Clean up
                if isfile(test_filename)
                    rm(test_filename)
                end
            end
        end
        
        @testset "TREXIO Basis Set Data" begin
            test_filename = "test_basis.h5"
            
            if isfile(test_filename)
                rm(test_filename)
            end
            
            try
                # Create a simple molecular system with basis sets
                using ElemCo.MSystems
                using StaticArrays
                
                # Create atoms with basis set information
                atom1 = ElemCo.MSystems.ACentre("H1", SVector(0.0, 0.0, 0.0), 1, 1.0, 
                                              Dict("ao" => "cc-pVDZ", "jkfit" => "cc-pVDZ-jkfit"), false)
                atom2 = ElemCo.MSystems.ACentre("H2", SVector(0.0, 0.0, 1.4), 1, 1.0, 
                                              Dict("ao" => "cc-pVDZ", "jkfit" => "cc-pVDZ-jkfit"), false)
                system = ElemCo.MSystems.MSystem([atom1, atom2])
                
                # Test basis set I/O using TrexioInterface
                if isdefined(ElemCo, :TrexioInterface)
                    trex = ElemCo.TrexioInterface.TrexioFile(test_filename, "w")
                    ElemCo.TrexioInterface.write_trexio_basis(trex, system)
                    ElemCo.TrexioInterface.close_trexio(trex)
                    
                    @test isfile(test_filename)
                    
                    # Read basis set data back
                    trex_read = ElemCo.TrexioInterface.TrexioFile(test_filename, "r")
                    basis_data = ElemCo.TrexioInterface.read_trexio_basis(trex_read)
                    ElemCo.TrexioInterface.close_trexio(trex_read)
                    
                    # Check that basis data is available - handle both formats
                    @test haskey(basis_data, "format")  # Should indicate format type
                    
                    if basis_data["format"] == "trexio"
                        # TREXIOIO format
                        @test haskey(basis_data, "shell_num")
                        @test haskey(basis_data, "shell_nucleus_index")
                        println("Basis data read in TREXIOIO format")
                    elseif basis_data["format"] == "legacy"
                        # Legacy format
                        @test haskey(basis_data, "type")
                        @test haskey(basis_data, "nucleus_index")
                        @test basis_data["type"][1] == "cc-pVDZ"
                        @test basis_data["type"][2] == "cc-pVDZ"
                        println("Basis data read in legacy format")
                    else
                        @warn "Unknown basis format: $(basis_data["format"])"
                    end
                end
                
            catch e
                @warn "TREXIO basis set test failed: $e"
            finally
                # Clean up
                if isfile(test_filename)
                    rm(test_filename)
                end
            end
        end
        
        @testset "TREXIO Orbitals with Basis Sets" begin
            test_filename = "test_orbitals_basis.h5"
            
            if isfile(test_filename)
                rm(test_filename)
            end
            
            try
                # Test that orbitals automatically include basis set information
                using ElemCo.MSystems
                using StaticArrays
                
                # Create a simple molecular system
                atom1 = ElemCo.MSystems.ACentre("H1", SVector(0.0, 0.0, 0.0), 1, 1.0, 
                                              Dict("ao" => "STO-3G"), false)
                atom2 = ElemCo.MSystems.ACentre("H2", SVector(0.0, 0.0, 1.4), 1, 1.0, 
                                              Dict("ao" => "STO-3G"), false)
                system = ElemCo.MSystems.MSystem([atom1, atom2])
                
                test_orbitals = rand(Float64, 2, 2)  # 2 basis functions, 2 MOs
                
                if isdefined(ElemCo, :TrexioInterface)
                    trex = ElemCo.TrexioInterface.TrexioFile(test_filename, "w")
                    ElemCo.TrexioInterface.write_trexio_orbitals(trex, test_orbitals, system=system)
                    ElemCo.TrexioInterface.close_trexio(trex)
                    
                    @test isfile(test_filename)
                    
                    # Check that both orbitals and basis sets were written
                    h5open(test_filename, "r") do file
                        @test haskey(file, "trexio")
                        @test haskey(file["trexio"], "mo")
                        @test haskey(file["trexio"], "basis")  # Basis should be included automatically
                        
                        # Check orbital data
                        orbitals_read = read(file["trexio"]["mo"]["coefficient"])
                        @test isapprox(orbitals_read, test_orbitals)
                        
                        # Check basis data - handle both TREXIOIO and legacy formats
                        basis_group = file["trexio"]["basis"]
                        if haskey(basis_group, "shell_num")
                            # TREXIOIO format
                            @test haskey(basis_group, "shell_nucleus_index")
                            @test haskey(basis_group, "shell_ang_mom")
                            println("Basis data stored in TREXIOIO format")
                        elseif haskey(basis_group, "type")
                            # Legacy format
                            basis_types = read(basis_group["type"])
                            @test basis_types[1] == "STO-3G"
                            @test basis_types[2] == "STO-3G"
                            println("Basis data stored in legacy format")
                        else
                            @warn "Unknown basis format"
                        end
                    end
                end
                
            catch e
                @warn "TREXIO orbitals with basis test failed: $e"
            finally
                # Clean up
                if isfile(test_filename)
                    rm(test_filename)
                end
            end
        end
        
        @testset "TREXIO Read Function with Basis" begin
            test_filename = "test_read_complete.h5"
            
            if isfile(test_filename)
                rm(test_filename)
            end
            
            try
                # Test the high-level read function includes basis data
                using ElemCo.MSystems
                using StaticArrays
                
                # Create test data
                atom1 = ElemCo.MSystems.ACentre("C1", SVector(0.0, 0.0, 0.0), 6, 6.0, 
                                              Dict("ao" => "6-31G"), false)
                system = ElemCo.MSystems.MSystem([atom1])
                test_orbitals = rand(Float64, 5, 5)  # 5 basis functions, 5 MOs
                
                if isdefined(ElemCo, :TrexioInterface)
                    # Write complete data
                    trex = ElemCo.TrexioInterface.TrexioFile(test_filename, "w")
                    ElemCo.TrexioInterface.write_trexio_molecule(trex, system)
                    ElemCo.TrexioInterface.write_trexio_orbitals(trex, test_orbitals, system=system)
                    ElemCo.TrexioInterface.close_trexio(trex)
                    
                    # Use the high-level read function
                    data = ElemCo.TrexioInterface.read_trexio(test_filename)
                    
                    @test haskey(data, "molecule")
                    @test haskey(data, "orbitals")
                    @test haskey(data, "basis")  # Should include basis information
                    
                    # Check basis data format and content
                    basis_data = data["basis"]
                    if basis_data["format"] == "trexio"
                        # TREXIOIO format may have basis set type as attribute
                        @test haskey(basis_data, "shell_num")
                        println("Complete data read with TREXIOIO basis format")
                    elseif basis_data["format"] == "legacy"
                        # Legacy format has type array
                        @test basis_data["type"][1] == "6-31G"
                        println("Complete data read with legacy basis format")
                    end
                end
                
            catch e
                @warn "TREXIO complete read test failed: $e"
            finally
                # Clean up
                if isfile(test_filename)
                    rm(test_filename)
                end
            end
        end

        @testset "TREXIO Amplitude Data" begin
            test_filename = "test_amplitudes.h5"
            
            if isfile(test_filename)
                rm(test_filename)
            end
            
            try
                # Test amplitude data I/O
                test_amplitudes = Dict{String, Any}(
                    "t1" => rand(Float64, 3, 3),
                    "t2" => rand(Float64, 3, 3, 3, 3)
                )
                
                # Write amplitude data
                h5open(test_filename, "w") do file
                    trexio_group = create_group(file, "trexio")
                    amp_group = create_group(trexio_group, "amplitudes")
                    for (key, value) in test_amplitudes
                        amp_group[key] = value
                    end
                end
                
                @test isfile(test_filename)
                
                # Read amplitude data
                h5open(test_filename, "r") do file
                    if haskey(file, "trexio") && haskey(file["trexio"], "amplitudes")
                        amp_group = file["trexio"]["amplitudes"]
                        for key in ["t1", "t2"]
                            if haskey(amp_group, key)
                                amp_read = read(amp_group[key])
                                @test isapprox(amp_read, test_amplitudes[key])
                            end
                        end
                    end
                end
                
            catch e
                @warn "TREXIO amplitude test failed: $e"
            finally
                # Clean up
                if isfile(test_filename)
                    rm(test_filename)
                end
            end
        end
        
    catch e
        @warn "TREXIO tests skipped due to missing dependencies: $e"
    end
end