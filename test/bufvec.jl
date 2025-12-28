using Test
using ElemCo
using ElemCo.Utils

# Test the new parametrized BufVec
println("Testing BufVec{T,A} implementation...")

@testset "BufVec Parametrized Construction" begin
    # Test with Vector
    data = Vector{Float64}(undef, 10)
    buf = BufVec(data)
    @test buf isa BufVec{Float64,Vector{Float64}}
    @test length(buf) == 0
    @test capacity(buf) == 10
    
    # Test with different element type
    data_int = Vector{Int}(undef, 5)
    buf_int = BufVec(data_int)
    @test buf_int isa BufVec{Int,Vector{Int}}
    @test eltype(buf_int) == Int
    
    # Test with initial length
    data2 = [1.0, 2.0, 3.0, 0.0, 0.0]
    buf2 = BufVec(data2, 3)
    @test length(buf2) == 3
    @test buf2[1] == 1.0
    @test buf2[3] == 3.0
end

@testset "BufVec No Auto-Grow" begin
    data = Vector{Int}(undef, 3)
    buf = BufVec(data)
    
    # Fill to capacity
    push!(buf, 1)
    push!(buf, 2)
    push!(buf, 3)
    @test length(buf) == 3
    @test capacity(buf) == 3
    @test is_full(buf)
    
    # Should error when full
    @test_throws ArgumentError push!(buf, 4)
end

@testset "BufVec SIMD Support" begin
    data = Vector{Float64}(undef, 100)
    buf = BufVec(data)
    
    # Fill with values
    for i in 1:50
        push!(buf, Float64(i))
    end
    
    # Test SIMD iteration
    function sum_with_simd(buf::BufVec{Float64})
        s = 0.0
        @inbounds @simd for i in eachindex(buf)
            s += buf[i]
        end
        return s
    end
    
    result = sum_with_simd(buf)
    expected = sum(1:50)
    @test result ≈ expected
end

@testset "BufVec Type Stability" begin
    data = Vector{Float64}(undef, 10)
    buf = BufVec(data)
    
    @inferred push!(buf, 1.0)
    @inferred buf[1]
    @inferred length(buf)
    @inferred capacity(buf)
end

@testset "BufVec Basic Operations" begin
    data = Vector{Int}(undef, 10)
    buf = BufVec(data)
    
    # Push and access
    push!(buf, 42)
    @test buf[1] == 42
    @test length(buf) == 1
    
    # Append
    append!(buf, [1, 2, 3])
    @test length(buf) == 4
    @test buf[2] == 1
    @test buf[4] == 3
    
    # Pop
    val = pop!(buf)
    @test val == 3
    @test length(buf) == 3
    
    # Empty
    empty!(buf)
    @test length(buf) == 0
    @test isempty(buf)
    @test capacity(buf) == 10  # Capacity unchanged
end

@testset "BufVec Iteration" begin
    data = Vector{Int}(undef, 10)
    buf = BufVec(data)
    append!(buf, [1, 2, 3, 4, 5])
    
    # Collect
    vec = collect(buf)
    @test vec == [1, 2, 3, 4, 5]
    
    # Manual iteration
    sum_val = 0
    for val in buf
        sum_val += val
    end
    @test sum_val == 15
end

@testset "BufVec Comparison and Copy" begin
    data1 = Vector{Int}(undef, 10)
    data2 = Vector{Int}(undef, 10)
    buf1 = BufVec(data1)
    buf2 = BufVec(data2)
    
    append!(buf1, [1, 2, 3])
    append!(buf2, [1, 2, 3])
    
    @test buf1 == buf2
    
    buf2[2] = 100
    @test buf1 != buf2
    
    # Copy
    buf3 = copy(buf1)
    @test buf3 == buf1
    @test buf3.data !== buf1.data
end

println("\n=== All BufVec parametrized tests passed! ===\n")
