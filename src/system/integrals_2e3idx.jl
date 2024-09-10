# 2-electron 3-index integrals
# adapted from GaussianBasis.jl

const INTEGRAL_NAMES_2E3IDX = (
  ("eri_2e3idx", "_", "two-electron three-index electron-repulsion"),
                              )

# automatically generate functions for the 2-electron 3-index integrals
for (jname_str, type, descr_str) in INTEGRAL_NAMES_2E3IDX
  jname = Symbol(jname_str)
  jname_ex = Symbol(jname_str*"!")
  descr = Symbol(descr_str)
  for (suffix, cartesian) in (("sph", false), ("cart", true))
    jname_sfx = Symbol(jname_str*"_$(suffix)")
    jname_sfx_ex = Symbol(jname_str*"_$(suffix)!")
    libname = Symbol("cint3c2e$(type)$(suffix)!")
    docstr = """
          $jname_sfx(ash1ao::AngularShell, ash2ao::AngularShell, ashfit::AngularShell, basis::BasisSet)

        Compute the $descr integral ``v_{a_1}^{a_2 P}`` for given angular shells.
        `basis` has to contain ao and fit bases.
      """
    docstr_ex = """
          $jname_ex(out, ash1ao::AngularShell, ash2ao::AngularShell, ashfit::AngularShell, basis::BasisSet)

        Compute the $descr integral ``v_{a_1}^{a_2 P}`` for given angular shells.
        `basis` has to contain ao and fit bases.
        The result is stored in `out`. 
      """
    docstr_sfx_ex = """
          $jname_sfx_ex(out, i::Int, j::Int, P::Int, basis::BasisSet)

        Compute the $descr integral ``v_i^{j P}`` for given angular shells.
        `basis` has to contain ao and fit bases.
        The result is stored in `out`. 
      """
    @eval begin
      @doc $docstr
      function $jname_sfx(ash1ao::AngularShell, ash2ao::AngularShell, ashfit::AngularShell, basis::BasisSet)
        buf = Array{Float64,3}(undef, n_ao(ash1ao,$cartesian),n_ao(ash2ao,$cartesian),n_ao(ashfit,$cartesian))
        $libname(buf, [ash1ao.id,ash2ao.id,ashfit.id], basis.lib)
        return buf
      end
        
      @doc $docstr_ex
      function $jname_sfx_ex(out, ash1ao::AngularShell, ash2ao::AngularShell, ashfit::AngularShell, basis::BasisSet)
        $libname(out, [ash1ao.id,ash2ao.id,ashfit.id], basis.lib)
      end

      @doc $docstr_sfx_ex
      function $jname_sfx_ex(out, i::Int, j::Int, P::Int, basis::BasisSet)
        $libname(out, [i,j,P], basis.lib)
      end
    end
  end
  jname_sph = Symbol(jname_str*"_sph!")
  jname_cart = Symbol(jname_str*"_cart!")
  docstr = """
        $jname(ao_basis::BasisSet, fit_basis::BasisSet)

      Compute the $descr integral.
    """
  docstr_ex = """
        $jname_ex(out, ao_basis::BasisSet, fit_basis::BasisSet)

      Compute the $descr integral.
      The result is stored in `out`. 
    """
  @eval begin
    @doc $docstr
    function $jname(ao_basis::BasisSet, fit_basis::BasisSet) 
      if is_cartesian(ao_basis) 
        @assert is_cartesian(fit_basis) "Basis sets must be both cartesian or both spherical"
        calc_2e3idx($jname_cart, ao_basis, fit_basis) 
      else
        calc_2e3idx($jname_sph, ao_basis, fit_basis)
      end
    end

    @doc $docstr_ex
    function $jname_ex(out, ao_basis::BasisSet, fit_basis::BasisSet) 
      if is_cartesian(ao_basis) 
        @assert is_cartesian(fit_basis) "Basis sets must be both cartesian or both spherical"
        calc_2e3idx!(out, $jname_cart, ao_basis, fit_basis)
      else
        calc_2e3idx!(out, $jname_sph, ao_basis, fit_basis)
      end
    end
  end
end

function calc_2e3idx(callback::Function, ao_basis::BasisSet, fit_basis::BasisSet)
  nao = n_ao(ao_basis)
  nfit = n_ao(fit_basis)
  out = zeros(nao, nao, nfit)
  calc_2e3idx!(out, callback, ao_basis, fit_basis)
end

function calc_2e3idx!(out, callback::Function, ao_basis::BasisSet, fit_basis::BasisSet)
  # Number of orbitals per shell
  nao4sh = Int[n_ao(ash, ao_basis.cartesian) for ash in ao_basis]
  nfit4sh = Int[n_ao(ash, fit_basis.cartesian) for ash in  fit_basis]
  nao_max = maximum(nao4sh)
  nfit_max = maximum(nfit4sh)

  bs = combine(ao_basis, fit_basis)

  # Offset list for each shell, used to map shell index to orbital index
  ao_offset = cumsum(vcat(0, nao4sh)) 
  fit_offset = cumsum(vcat(0, nfit4sh))

  buf_arrays = [zeros(Cdouble, nao_max^2*nfit_max) for _ = 1:Threads.nthreads()]

  @sync for (P, Pb) in enumerate(shell_range(bs,2))
    Threads.@spawn begin
      @inbounds begin
        buf = buf_arrays[Threads.threadid()]
        nP = nfit4sh[P]
        Pblk = (1:nP) .+ fit_offset[P]
        for (j, jb) in enumerate(shell_range(bs,1))
          nj = nao4sh[j]
          jblk = (1:nj) .+ ao_offset[j]
          for (i, ib) in enumerate(shell_range(bs,1).start:jb) # Only upper triangle
            ni = nao4sh[i]
            iblk = (1:ni) .+ ao_offset[i]

            # Call libcint
            callback(buf, ib, jb, Pb, bs)
            
            # save elements
            vbuf = reshape_buf(buf, ni, nj, nP)
            out[iblk, jblk, Pblk] = vbuf
            v_jiP = @view out[jblk, iblk, Pblk]
            permutedims!(v_jiP, vbuf, (2,1,3))
          end
        end
      end #inbounds
    end #spwan
  end #sync
  return out
end