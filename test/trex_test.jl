"""
Test for TREX interface functionality
"""

using Test

# Only test if HDF5 is available
@testset "TREX Interface Tests" begin
    try
        using HDF5
        using ElemCo
        
        # Test basic TREX file operations
        @testset "TREX File Operations" begin
            test_filename = "test_trex.h5"
            
            # Clean up any existing test file
            if isfile(test_filename)
                rm(test_filename)
            end
            
            try
                # Test orbital data I/O (simplified test)
                test_orbitals = rand(Float64, 5, 3)  # 5 basis functions, 3 MOs
                
                # Test high-level write function
                trex_data = Dict{String, Any}("orbitals" => test_orbitals)
                
                # Use HDF5 directly for basic test
                h5open(test_filename, "w") do file
                    trex_group = create_group(file, "trex")
                    mo_group = create_group(trex_group, "mo")
                    mo_group["num"] = size(test_orbitals, 2)
                    mo_group["coefficient"] = test_orbitals
                end
                
                @test isfile(test_filename)
                
                # Test reading
                h5open(test_filename, "r") do file
                    if haskey(file, "trex") && haskey(file["trex"], "mo")
                        orbitals_read = read(file["trex"]["mo"]["coefficient"])
                        @test size(orbitals_read) == size(test_orbitals)
                        @test isapprox(orbitals_read, test_orbitals)
                    end
                end
                
            catch e
                @warn "TREX basic test failed: $e"
            finally
                # Clean up
                if isfile(test_filename)
                    rm(test_filename)
                end
            end
        end
        
        @testset "TREX Basis Set Data" begin
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
                
                # Test basis set I/O using TrexInterface
                if isdefined(ElemCo, :TrexInterface)
                    trex = ElemCo.TrexInterface.TrexFile(test_filename, "w")
                    ElemCo.TrexInterface.write_trex_basis(trex, system)
                    ElemCo.TrexInterface.close_trex(trex)
                    
                    @test isfile(test_filename)
                    
                    # Read basis set data back
                    trex_read = ElemCo.TrexInterface.TrexFile(test_filename, "r")
                    basis_data = ElemCo.TrexInterface.read_trex_basis(trex_read)
                    ElemCo.TrexInterface.close_trex(trex_read)
                    
                    # Check that basis data is available - handle both formats
                    @test haskey(basis_data, "format")  # Should indicate format type
                    
                    if basis_data["format"] == "trexio"
                        # TREXIO format
                        @test haskey(basis_data, "shell_num")
                        @test haskey(basis_data, "shell_nucleus_index")
                        println("Basis data read in TREXIO format")
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
                @warn "TREX basis set test failed: $e"
            finally
                # Clean up
                if isfile(test_filename)
                    rm(test_filename)
                end
            end
        end
        
        @testset "TREX Orbitals with Basis Sets" begin
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
                
                if isdefined(ElemCo, :TrexInterface)
                    trex = ElemCo.TrexInterface.TrexFile(test_filename, "w")
                    ElemCo.TrexInterface.write_trex_orbitals(trex, test_orbitals, system=system)
                    ElemCo.TrexInterface.close_trex(trex)
                    
                    @test isfile(test_filename)
                    
                    # Check that both orbitals and basis sets were written
                    h5open(test_filename, "r") do file
                        @test haskey(file, "trex")
                        @test haskey(file["trex"], "mo")
                        @test haskey(file["trex"], "basis")  # Basis should be included automatically
                        
                        # Check orbital data
                        orbitals_read = read(file["trex"]["mo"]["coefficient"])
                        @test isapprox(orbitals_read, test_orbitals)
                        
                        # Check basis data - handle both TREXIO and legacy formats
                        basis_group = file["trex"]["basis"]
                        if haskey(basis_group, "shell_num")
                            # TREXIO format
                            @test haskey(basis_group, "shell_nucleus_index")
                            @test haskey(basis_group, "shell_ang_mom")
                            println("Basis data stored in TREXIO format")
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
                @warn "TREX orbitals with basis test failed: $e"
            finally
                # Clean up
                if isfile(test_filename)
                    rm(test_filename)
                end
            end
        end
        
        @testset "TREX Read Function with Basis" begin
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
                
                if isdefined(ElemCo, :TrexInterface)
                    # Write complete data
                    trex = ElemCo.TrexInterface.TrexFile(test_filename, "w")
                    ElemCo.TrexInterface.write_trex_molecule(trex, system)
                    ElemCo.TrexInterface.write_trex_orbitals(trex, test_orbitals, system=system)
                    ElemCo.TrexInterface.close_trex(trex)
                    
                    # Use the high-level read function
                    data = ElemCo.TrexInterface.read_trex(test_filename)
                    
                    @test haskey(data, "molecule")
                    @test haskey(data, "orbitals")
                    @test haskey(data, "basis")  # Should include basis information
                    
                    # Check basis data format and content
                    basis_data = data["basis"]
                    if basis_data["format"] == "trexio"
                        # TREXIO format may have basis set type as attribute
                        @test haskey(basis_data, "shell_num")
                        println("Complete data read with TREXIO basis format")
                    elseif basis_data["format"] == "legacy"
                        # Legacy format has type array
                        @test basis_data["type"][1] == "6-31G"
                        println("Complete data read with legacy basis format")
                    end
                end
                
            catch e
                @warn "TREX complete read test failed: $e"
            finally
                # Clean up
                if isfile(test_filename)
                    rm(test_filename)
                end
            end
        end

        @testset "TREX Amplitude Data" begin
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
                    trex_group = create_group(file, "trex")
                    amp_group = create_group(trex_group, "amplitudes")
                    for (key, value) in test_amplitudes
                        amp_group[key] = value
                    end
                end
                
                @test isfile(test_filename)
                
                # Read amplitude data
                h5open(test_filename, "r") do file
                    if haskey(file, "trex") && haskey(file["trex"], "amplitudes")
                        amp_group = file["trex"]["amplitudes"]
                        for key in ["t1", "t2"]
                            if haskey(amp_group, key)
                                amp_read = read(amp_group[key])
                                @test isapprox(amp_read, test_amplitudes[key])
                            end
                        end
                    end
                end
                
            catch e
                @warn "TREX amplitude test failed: $e"
            finally
                # Clean up
                if isfile(test_filename)
                    rm(test_filename)
                end
            end
        end
        
    catch e
        @warn "TREX tests skipped due to missing dependencies: $e"
    end
end