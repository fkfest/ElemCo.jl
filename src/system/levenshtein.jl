"""
    levenshtein_distance(s1::AbstractString, s2::AbstractString)

Calculate the Levenshtein distance between two strings.
The Levenshtein distance is the minimum number of single-character edits 
(insertions, deletions, or substitutions) required to change one string into another.

Based on the algorithm suggested in https://github.com/rawrgrr/Levenshtein.jl
"""
function levenshtein_distance(s1::AbstractString, s2::AbstractString)
    len1, len2 = length(s1), length(s2)
    
    # Create a matrix to store distances
    dist = Matrix{Int}(undef, len1 + 1, len2 + 1)
    
    # Initialize first row and column
    for i in 1:(len1 + 1)
        dist[i, 1] = i - 1
    end
    for j in 1:(len2 + 1)
        dist[1, j] = j - 1
    end
    
    # Fill the matrix
    for i in 2:(len1 + 1)
        for j in 2:(len2 + 1)
            cost = s1[i-1] == s2[j-1] ? 0 : 1
            dist[i, j] = min(
                dist[i-1, j] + 1,     # deletion
                dist[i, j-1] + 1,     # insertion
                dist[i-1, j-1] + cost # substitution
            )
        end
    end
    
    return dist[len1 + 1, len2 + 1]
end

"""
    get_available_basis_sets()

Get a list of all available basis set names from the basis library.
Returns basis set names without version suffixes and file extensions.
"""
function get_available_basis_sets()
    basis_dir = joinpath(@__DIR__, "..", "..", "lib", "basis_sets", "mpro")
    basis_files = readdir(basis_dir)
    
    # Extract basis set names (remove .X.mpro suffixes)
    basis_names = Set{String}()
    for file in basis_files
        if endswith(file, ".mpro")
            # Remove version and extension: "cc-pvdz.0.mpro" -> "cc-pvdz"
            name = replace(file, r"\.\d+\.mpro$" => "")
            push!(basis_names, name)
        end
    end
    
    return sort(collect(basis_names))
end

"""
    suggest_basis_sets(target::AbstractString, max_suggestions::Int=5, max_distance::Int=3)

Suggest basis set names that are similar to the target string using Levenshtein distance.
Returns up to `max_suggestions` suggestions with distance ≤ `max_distance`, sorted by distance.
"""
function suggest_basis_sets(target::AbstractString, max_suggestions::Int=5, max_distance::Int=3)
    available_sets = get_available_basis_sets()
    target_lower = lowercase(target)
    
    # Calculate distances and filter
    suggestions = Tuple{String, Int}[]
    for basis_name in available_sets
        distance = levenshtein_distance(target_lower, lowercase(basis_name))
        if distance <= max_distance
            push!(suggestions, (basis_name, distance))
        end
    end
    
    # Sort by distance (closest first) and limit results
    sort!(suggestions, by = x -> x[2])
    return [name for (name, _) in suggestions[1:min(max_suggestions, length(suggestions))]]
end