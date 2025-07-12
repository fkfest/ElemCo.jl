using Test
using ElemCo

@testset "Levenshtein Distance and Basis Set Suggestions" begin
    
    # Test levenshtein_distance function
    @testset "levenshtein_distance" begin
        @test ElemCo.BasisSets.levenshtein_distance("", "") == 0
        @test ElemCo.BasisSets.levenshtein_distance("a", "a") == 0
        @test ElemCo.BasisSets.levenshtein_distance("", "a") == 1
        @test ElemCo.BasisSets.levenshtein_distance("a", "") == 1
        @test ElemCo.BasisSets.levenshtein_distance("kitten", "sitting") == 3
        @test ElemCo.BasisSets.levenshtein_distance("cc-pvdz", "cc-pvtz") == 1
        @test ElemCo.BasisSets.levenshtein_distance("def2-svp", "def2-tzvp") == 2
    end
    
    # Test get_available_basis_sets function
    @testset "get_available_basis_sets" begin
        basis_sets = ElemCo.BasisSets.get_available_basis_sets()
        @test isa(basis_sets, Vector{String})
        @test length(basis_sets) > 0
        @test "cc-pvdz" in basis_sets
        @test "def2-svp" in basis_sets
        # Check that basis sets are sorted
        @test issorted(basis_sets)
    end
    
    # Test suggest_basis_sets function
    @testset "suggest_basis_sets" begin
        suggestions = ElemCo.BasisSets.suggest_basis_sets("cc-pvd")
        @test isa(suggestions, Vector{String})
        @test "cc-pvdz" in suggestions  # Should be close match
        
        suggestions = ElemCo.BasisSets.suggest_basis_sets("def2sv")
        @test "def2-svp" in suggestions  # Should be close match
        
        # Test with max_suggestions parameter
        suggestions = ElemCo.BasisSets.suggest_basis_sets("cc", 2)
        @test length(suggestions) <= 2
        
        # Test with very different string - should return fewer or no results
        suggestions = ElemCo.BasisSets.suggest_basis_sets("xyz123")
        @test length(suggestions) <= 5  # Should respect default max_suggestions
    end
    
    # Test integration with parse_basis error handling
    @testset "Error message with suggestions" begin
        # This should throw an error with suggestions
        @test_throws ErrorException ElemCo.BasisSets.parse_basis("cc-pvd", ElemCo.Elements.ACentre("H", [0.0, 0.0, 0.0]))
        
        # Capture the error message to verify it contains suggestions
        try
            ElemCo.BasisSets.parse_basis("cc-pvd", ElemCo.Elements.ACentre("H", [0.0, 0.0, 0.0]))
        catch e
            @test occursin("Did you mean", e.msg)
            @test occursin("cc-pvdz", e.msg)
        end
    end
end