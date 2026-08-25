@testitem "ciphi_uhf" tags=[:ciphi, :quick] begin
using Test
using ElemCo
using ElemCo.TrexioInterface

@testset "CIPHI - UHF Systems" begin
  println("\n=== Testing CIPHI with UHF ===")
    
  epsilon = 1.e-6
  E_CIPHIa_test = -75.940790606682
  E_CIPHIa_PT2_test = -75.941009904381
  E_CIPHIc_test = -75.688047789381
  E_CIPHIt_test = -75.852590484068

  E_CIPHIa_tight_test = -75.941001448353

  E_CIPHIa_ms_test = -75.939226562784
  omega1_test = 0.08963377462914934
    
  @testset "CIPHI Basic - H2O Anion" begin
        
    geometry = "
               O      0.000000000    0.000000000   -0.130186067
               H1     0.000000000    1.489124508    1.033245507
               H2     0.000000000   -1.489124508    1.033245507"
    basis = Dict("ao"=>"6-31g", "jkfit"=>"vtz-jkfit", "mpfit"=>"vtz-mpfit")
    
    @set wf charge=-1
    @set ciphi epsilon=1.e-4
    
    @dfuhf
    energies = @ciphi
    
    @test haskey(energies, "CIPHI")
    E_ciphi = energies["CIPHI"]
    
    println("CIPHI Energy (H2O anion, UHF): $E_ciphi")
    @test abs(E_ciphi - E_CIPHIa_test) < epsilon
  end
    
  @testset "CIPHI - H2O Cation" begin
    
    geometry = "
               O      0.000000000    0.000000000   -0.130186067
               H1     0.000000000    1.489124508    1.033245507
               H2     0.000000000   -1.489124508    1.033245507"
    basis = Dict("ao"=>"6-31g", "jkfit"=>"vtz-jkfit", "mpfit"=>"vtz-mpfit")
    
    @set wf charge=1

    @dfuhf
    energies = @ciphi
    
    E_ciphi = energies["CIPHI"]
    
    println("CIPHI Energy (H2O cation, UHF): $E_ciphi")
    @test abs(E_ciphi - E_CIPHIc_test) < epsilon
  end

  @testset "CIPHI Properties - H2O Cation" begin
    geometry = "
               O      0.000000000    0.000000000   -0.130186067
               H1     0.000000000    1.489124508    1.033245507
               H2     0.000000000   -1.489124508    1.033245507"
    basis = Dict("ao"=>"6-31g", "jkfit"=>"vtz-jkfit", "mpfit"=>"vtz-mpfit")

    ciphi_natorb = "ciphi_uhf_natorb.h5"
    @set wf charge=1
    @set ciphi properties=true
    @set wf natorb=ciphi_natorb

    @dfuhf
    energies = @ciphi

    @test abs(energies["CIPHI"] - E_CIPHIc_test) < epsilon
    @test abs(energies["DMZ"]) > 0.0
    open_trexio(joinpath(EC.scr, ciphi_natorb), "r") do io
      @test !ElemCo.TREXIO.trexio_has_rdm_1e(io)
      @test !ElemCo.TREXIO.trexio_has_rdm_1e_up(io)
      @test !ElemCo.TREXIO.trexio_has_rdm_1e_dn(io)
      occa, occb = read_trexio_orbital_occupations(io, "mo")
      @test abs(sum(occa) - 5.0) < epsilon
      @test abs(sum(occb) - 4.0) < epsilon
    end
    open_trexio(ElemCo.Wavefunctions.dumpfile(EC, "w")[2], "r") do io
      @test ElemCo.TREXIO.trexio_has_rdm_1e(io)
      @test ElemCo.TREXIO.trexio_has_rdm_1e_up(io)
      @test ElemCo.TREXIO.trexio_has_rdm_1e_dn(io)
    end
    @set ciphi properties=false
    @set wf natorb=""
  end
    
  @testset "CIPHI - H2O Triplet" begin
    geometry = "
               O      0.000000000    0.000000000   -0.130186067
               H1     0.000000000    1.489124508    1.033245507
               H2     0.000000000   -1.489124508    1.033245507"
    basis = Dict("ao"=>"6-31g", "jkfit"=>"vtz-jkfit", "mpfit"=>"vtz-mpfit")

    @set wf ms2=2
    
    @dfuhf
    energies = @ciphi
    
    E_ciphi = energies["CIPHI"]
    
    println("CIPHI Energy (H2O triplet, UHF): $E_ciphi")
    @test abs(E_ciphi - E_CIPHIt_test) < epsilon
  end
    
  @testset "CIPHI UHF Selection Thresholds" begin
        
    geometry = "
               O      0.000000000    0.000000000   -0.130186067
               H1     0.000000000    1.489124508    1.033245507
               H2     0.000000000   -1.489124508    1.033245507"
    basis = Dict("ao"=>"6-31g", "jkfit"=>"vtz-jkfit", "mpfit"=>"vtz-mpfit")
    
    @set wf charge=-1
    @set ciphi epsilon=1.e-5
    
    @dfuhf
    energies = @ciphi
    
    E_ciphi = energies["CIPHI"]
    
    println("CIPHI Energy (UHF, tight selection): $E_ciphi")
    @test abs(E_ciphi - E_CIPHIa_tight_test) < epsilon
  end
    
  @testset "CIPHI UHF Multi-root" begin
    geometry = "
               O      0.000000000    0.000000000   -0.130186067
               H1     0.000000000    1.489124508    1.033245507
               H2     0.000000000   -1.489124508    1.033245507"
    basis = Dict("ao"=>"6-31g", "jkfit"=>"vtz-jkfit", "mpfit"=>"vtz-mpfit")
    
    @set wf charge=-1
    @set ciphi nstates=2 epsilon=5.e-4
    
    @dfuhf
    energies = @ciphi
    
    println("CIPHI Multi-root energies (UHF): ", energies["CIPHI"])
    
    @test abs(energies["CIPHI"] - E_CIPHIa_ms_test) < epsilon
    @test abs(energies["ω1"] - omega1_test) < epsilon
  end

  println("\n=== CIPHI UHF Tests Passed ===\n")
end
end
