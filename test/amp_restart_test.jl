@testitem "amp_restart" tags=[:cc, :quick] begin
using ElemCo
using Test

# Common geometry and basis for all tests
geometry = "bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"

basis = Dict("ao"=>"cc-pVDZ", "jkfit"=>"cc-pvtz-jkfit", "mpfit"=>"cc-pvdz-mpfit")

"""
    test_restart(name, first_calc, second_calc; epsilon=1.e-6, compare_key=nothing)

Test restart functionality by running first_calc, storing results, then running second_calc
with restart and comparing energies.
"""
function test_restart(name, first_calc::Function, second_calc::Function; 
                      epsilon=1.e-6, compare_key=nothing)
  tmpdir = mktempdir()
  store_file = joinpath(tmpdir, "newwf.h5")
  println("="^60)
  println("$name: First run")
  println("="^60)
  E1, key = first_calc(store_file)
  
  if isnothing(compare_key)
    compare_key = key
  end
  
  @test isfile(store_file)
  
  println("="^60)
  println("$name: Restart run")
  println("="^60)
  E2, _ = second_calc(store_file)
  
  println("$name energy (first): $E1")
  println("$name energy (restart): $E2")
  println("Difference: $(abs(E1 - E2))")
  
  @test abs(E1 - E2) < epsilon
  rm(tmpdir; force=true, recursive=true)
end

# =============================================================================
# DF-HF Restart Tests
# =============================================================================

@testset "DF-HF Restart" begin
  test_restart("DF-HF",
    (store_file) -> begin
      @set wf dump = store_file
      @set wf store = store_file
      energies = @dfhf
      (energies["HF"], "HF")
    end,
    (store_file) -> begin
      @set wf dump = store_file
      @set wf start = store_file
      @set scf maxit = 3  # Should converge quickly from restart
      energies = @dfhf
      (energies["HF"], "HF")
    end;
    epsilon=1.e-10
  )
end

@testset "DF-UHF Restart (anion)" begin
  test_restart("DF-UHF",
    (store_file) -> begin
      @set wf charge = -1 ms2 = 1
      @set wf dump = store_file
      @set wf store = store_file
      energies = @dfuhf
      (energies["HF"], "HF")
    end,
    (store_file) -> begin
      @set wf charge = -1 ms2 = 1
      @set wf dump = store_file
      @set wf start = store_file
      @set scf maxit = 3
      energies = @dfuhf
      (energies["HF"], "HF")
    end;
    epsilon=1.e-10
  )
end

# =============================================================================
# Restricted CC Restart Tests (with DF-HF orbitals)
# =============================================================================

@testset "DCSD Restart" begin
  test_restart("DCSD",
    (store_file) -> begin
      @set wf dump = store_file
      @set wf store = store_file
      @dfhf
      energies = @cc dcsd
      (energies["DCSD"], "DCSD")
    end,
    (store_file) -> begin
      @set wf dump = store_file
      @set wf store = store_file
      @dfhf
      energies = @cc dcsd begin
        wf(start=store_file)
      end
      (energies["DCSD"], "DCSD")
    end
  )
end

@testset "DCD Restart" begin
  test_restart("DCD",
    (store_file) -> begin
      @set wf dump = store_file
      @set wf store = store_file
      @dfhf
      energies = @cc dcd
      (energies["DCD"], "DCD")
    end,
    (store_file) -> begin
      @set wf dump = store_file
      @set wf start = store_file
      @set wf store = store_file
      @dfhf
      energies = @cc dcd
      (energies["DCD"], "DCD")
    end
  )
end

# =============================================================================
# Unrestricted CC Restart Tests
# =============================================================================

@testset "UDCSD Restart (anion)" begin
  test_restart("UDCSD",
    (store_file) -> begin
      @set wf charge = -1 ms2 = 1
      @dfuhf
      energies = @cc udcsd begin
        wf(store=store_file)
      end
      (energies["UDCSD"], "UDCSD")
    end,
    (store_file) -> begin
      @set wf charge = -1 ms2 = 1
      @dfuhf
      energies = @cc udcsd begin
        wf(start=store_file)
      end
      (energies["UDCSD"], "UDCSD")
    end
  )
end

@testset "UDCD Restart (anion)" begin
  test_restart("UDCD",
    (store_file) -> begin
      @set wf charge=-1 ms2=1 dump=store_file
      @dfuhf 
      energies = @cc udcd begin
        wf(store=store_file)
      end
      (energies["UDCD"], "UDCD")
    end,
    (store_file) -> begin
      @set wf charge=-1 ms2=1 
      @dfuhf begin
        wf(start=store_file)
      end
      energies = @cc udcd begin
        wf(start=store_file)
      end
      (energies["UDCD"], "UDCD")
    end
  )
end

# =============================================================================
# FCIDUMP-only Restart Tests (no molecular system, unity rotation stored)
# =============================================================================

@testset "FCIDUMP-only CCSD Restart" begin
  geometry = nothing 
  fcidump = joinpath(@__DIR__, "files", "H2O.FCIDUMP")
  
  test_restart("CCSD (FCIDUMP-only)",
    (store_file) -> begin
      @set wf store = store_file
      energies = @cc ccsd
      (energies["CCSD"], "CCSD")
    end,
    (store_file) -> begin
      @set wf start = store_file
      @set wf store = store_file
      energies = @cc ccsd
      (energies["CCSD"], "CCSD")
    end
  )
end

@testset "FCIDUMP-only DCD Restart" begin
  geometry = nothing
  fcidump = joinpath(@__DIR__, "files", "H2O.FCIDUMP")
  
  test_restart("DCD (FCIDUMP-only)",
    (store_file) -> begin
      energies = @cc dcd begin
        @set wf store = store_file
      end
      (energies["DCD"], "DCD")
    end,
    (store_file) -> begin
      energies = @cc dcd begin
        @set wf start=store_file store=store_file
      end
      (energies["DCD"], "DCD")
    end
  )
end

@testset "FCIDUMP-only UDCD Restart" begin
  geometry = nothing
  fcidump = joinpath(@__DIR__, "files", "H2O.FCIDUMP")
  test_restart("UDCD (FCIDUMP-only)",
    (store_file) -> begin
      @set wf charge=-1 store=store_file
      energies = @cc udcd
      (energies["UDCD"], "UDCD")
    end,
    (store_file) -> begin
      @set wf charge=-1 start=store_file store=store_file
      energies = @cc udcd
      (energies["UDCD"], "UDCD")
    end
  )
end

@testset "FCIDUMP-only UDCSD Restart" begin
  geometry = nothing
  fcidump = joinpath(@__DIR__, "files", "H2O.FCIDUMP")
  test_restart("UDCSD (FCIDUMP-only)",
    (store_file) -> begin
      @set wf charge=-1 store=store_file
      energies = @cc udcsd
      (energies["UDCSD"], "UDCSD")
    end,
    (store_file) -> begin
      @set wf charge=-1 start=store_file store=store_file
      energies = @cc udcsd
      (energies["UDCSD"], "UDCSD")
    end
  )
end
end
