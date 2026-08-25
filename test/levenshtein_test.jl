@testitem "levenshtein" tags=[:df, :quick] begin
using Test

# Test for Levenshtein distance and basis set suggestions functionality
@testset "Levenshtein Distance and Basis Set Suggestions" begin
    
    # Simple test of the functionality by including the file directly
    levenshtein_file = joinpath(@__DIR__, "..", "src", "system", "levenshtein.jl")
    include(levenshtein_file)
    
    # Test levenshtein_distance function
    @testset "levenshtein_distance" begin
        @test levenshtein_distance("", "") == 0
        @test levenshtein_distance("a", "a") == 0
        @test levenshtein_distance("", "a") == 1
        @test levenshtein_distance("a", "") == 1
        @test levenshtein_distance("kitten", "sitting") == 3
        @test levenshtein_distance("cc-pvdz", "cc-pvtz") == 1
        @test levenshtein_distance("def2-svp", "def2-tzvp") == 2
    end
    
    # Test get_available_basis_sets function
    @testset "get_available_basis_sets" begin
        basis_sets = get_available_basis_sets()
        @test isa(basis_sets, Vector{String})
        @test length(basis_sets) > 0
        @test "cc-pvdz" in basis_sets
        @test "def2-svp" in basis_sets
        # Check that basis sets are sorted
        @test issorted(basis_sets)
    end
    
    # Test suggest_basis_sets function
    @testset "suggest_basis_sets" begin
        suggestions = suggest_basis_sets("cc-pvd")
        @test isa(suggestions, Vector{String})
        @test "cc-pvdz" in suggestions  # Should be close match
        
        suggestions = suggest_basis_sets("def2sv")
        @test "def2-svp" in suggestions  # Should be close match
        
        # Test with max_suggestions parameter
        suggestions = suggest_basis_sets("cc", 2)
        @test length(suggestions) <= 2
        
        # Test with very different string - should return fewer or no results
        suggestions = suggest_basis_sets("xyz123")
        @test length(suggestions) <= 5  # Should respect default max_suggestions
    end
    
    # Test that error message construction works correctly
    @testset "Error message formatting" begin
        # Test the error message construction logic
        function test_error_message(basis_name::String)
            suggestions = suggest_basis_sets(basis_name)
            if !isempty(suggestions)
                suggestion_str = join(suggestions, ", ")
                return "Basis set $basis_name not found! Did you mean: $suggestion_str?"
            else
                return "Basis set $basis_name not found!"
            end
        end
        
        msg1 = test_error_message("cc-pvd")
        @test occursin("Did you mean", msg1)
        @test occursin("cc-pvdz", msg1)
        
        msg2 = test_error_message("xyz123invalid")
        @test !occursin("Did you mean", msg2)
        @test occursin("not found!", msg2)
    end
end
end
