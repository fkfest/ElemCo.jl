# 2-index integrals (1-electron and 2-electron)
# adapted from GaussianBasis.jl

const INTEGRAL_NAMES_2IDX = ( 
  ("overlap", "1e_ovlp", "overlap"),
  ("kinetic", "1e_kin", "kinetic"),
  ("nuclear", "1e_nuc", "nuclear"),
  ("eri_2e2idx", "2c2e", "two-electron 2-index electron-repulsion") 
                            )

# automatically generate functions for the 1-electron integrals
for (jname_str, type, descr_str) in INTEGRAL_NAMES_2IDX
  jname = Symbol(jname_str)
  jname_ex = Symbol(jname_str*"!")
  descr = Symbol(descr_str)
  for (suffix,cartesian) in (("sph",false), ("cart",true))
    jname_sfx = Symbol(jname_str*"_$(suffix)")
    jname_sfx_ex = Symbol(jname_str*"_$(suffix)!")
    libname = Symbol("cint$(type)_$(suffix)!")
    docstr = """
          $jname_sfx(ash1::AngularShell, ash2::AngularShell, basis::BasisSet)

        Compute the $descr integral between two angular shells.
      """
    docstr_ex = """
          $jname_ex(out, ash1::AngularShell, ash2::AngularShell, basis::BasisSet)

        Compute the $descr integral between two angular shells.
        The result is stored in `out`. 
      """
    docstr_sfx_ex = """
          $jname_sfx_ex(out, i::Int, j::Int, basis::BasisSet)

        Compute the $descr integral between two angular shells.
        The result is stored in `out`. 
      """
    @eval begin
      @doc $docstr
      function $jname_sfx(ash1::AngularShell, ash2::AngularShell, basis::BasisSet)
        buf = Matrix{Float64}(undef, n_ao(ash1,$cartesian),n_ao(ash2,$cartesian))
        $libname(buf, [ash1.id,ash2.id], basis.lib)
        return buf
      end
        
      @doc $docstr_ex
      function $jname_sfx_ex(out, ash1::AngularShell, ash2::AngularShell, basis::BasisSet)
        $libname(out, [ash1.id,ash2.id], basis.lib)
      end

      @doc $docstr_sfx_ex
      function $jname_sfx_ex(out, i::Int, j::Int, basis::BasisSet)
        $libname(out, [i,j], basis.lib)
      end
    end
  end
  jname_sph = Symbol(jname_str*"_sph!")
  jname_cart = Symbol(jname_str*"_cart!")
  docstr = """
        $jname(basis::BasisSet)

      Compute the $descr integral matrix.
    """
  docstr_ex = """
        $jname_ex(out, basis::BasisSet)

      Compute the $descr integral matrix.
      The result is stored in `out`. 
    """
  @eval begin
    @doc $docstr
    function $jname(basis::BasisSet) 
      if is_cartesian(basis) 
        calc_1e($jname_cart, basis) 
      else
        calc_1e($jname_sph, basis)
      end
    end
    function $jname(basis1::BasisSet, basis2::BasisSet) 
      if is_cartesian(basis1) 
        @assert is_cartesian(basis2) "Basis sets must be both cartesian or both spherical"
        calc_1e($jname_cart, basis1, basis2) 
      else
        calc_1e($jname_sph, basis1, basis2)
      end
    end

    @doc $docstr_ex
    function $jname_ex(out, basis::BasisSet) 
      if is_cartesian(basis) 
        calc_1e!(out, $jname_cart, basis)
      else
        calc_1e!(out, $jname_sph, basis)
      end
    end
    function $jname_ex(out, basis1::BasisSet, basis2::BasisSet) 
      if is_cartesian(basis1)
        @assert is_cartesian(basis2) "Basis sets must be both cartesian or both spherical"
        calc_1e!(out, $jname_cart, basis1, basis2)
      else
        calc_1e!(out, $jname_sph, basis1, basis2)
      end
    end
  end
end

function calc_1e(callback::Function, bs::BasisSet)
  nao = n_ao(bs)
  out = zeros(nao, nao)
  calc_1e!(out, callback, bs)
  return out
end

function calc_1e!(out, callback::Function, bs::BasisSet)
  # Number of AOs per shell
  nao4sh = Int[n_ao(ash, bs.cartesian) for ash in bs]
  nao_max = maximum(nao4sh)

  # Offset list for each shell, used to map shell index to AO index
  ao_offset = cumsum(vcat(0, nao4sh)) 

  tbufs = ThreadsBuffer{Cdouble}(nao_max^2)

  @sync for (j, lenj) in enumerate(nao4sh)
    Threads.@spawn begin
      @inbounds begin
        buf = neuralyze(reshape_buf!(tbufs, length(tbufs)))
        joff = ao_offset[j]
        for i in 1:j
          leni = nao4sh[i]
          ioff = ao_offset[i]
          # Call libcint
          callback(buf, i, j, bs)
          # save elements
          vbuf = reshape_buf!(tbufs, leni, lenj)
          out[ioff+1:ioff+leni, joff+1:joff+lenj] = vbuf
          out[joff+1:joff+lenj, ioff+1:ioff+leni] = vbuf'
        end
        reset!(tbufs)
      end #inbounds
    end #spawn
  end #sync
  return out
end

function calc_1e(callback::Function, bs1::BasisSet, bs2::BasisSet)
  nao1 = n_ao(bs1)
  nao2 = n_ao(bs2)
  out = zeros(nao1, nao2)
  calc_1e!(out, callback, bs1, bs2)
  return out
end

function calc_1e!(out, callback::Function, bs1::BasisSet, bs2::BasisSet)
  # Number of AOs per shell
  nao4sh1 = Int[n_ao(ash, bs1.cartesian) for ash in bs1]
  nao4sh2 = Int[n_ao(ash, bs2.cartesian) for ash in bs2]
  nao_max1 = maximum(nao4sh1)
  nao_max2 = maximum(nao4sh2)

  bs = combine(bs1, bs2)

  # Offset list for each shell, used to map shell index to AO index
  ao_offset1 = cumsum(vcat(0, nao4sh1)) 
  ao_offset2 = cumsum(vcat(0, nao4sh2))

  tbufs = ThreadsBuffer{Cdouble}(nao_max1*nao_max2)

  @sync for (j, jb) in enumerate(shell_range(bs,2))
    Threads.@spawn begin
      @inbounds begin
        buf = neuralyze(reshape_buf!(tbufs, length(tbufs)))
        lenj = nao4sh2[j]
        joff = ao_offset2[j]
        for (i, ib) in enumerate(shell_range(bs,1))
          leni = nao4sh1[i]
          ioff = ao_offset1[i]

          # Call libcint
          callback(buf, ib, jb, bs)

          # save elements
          vbuf = reshape_buf!(tbufs, leni, lenj)
          out[ioff+1:ioff+leni, joff+1:joff+lenj] = vbuf
        end
        reset!(tbufs)
      end #inbounds
    end #spawn
  end #sync
  return out
end