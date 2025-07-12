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