using ElemCo
zeta = ["d","t","q","5"]
for z in zeta
  Z = uppercase(z)
  original = "v$(z)z-jkfit"  
  newfile = "aug-cc-pv$(z)z-jkfit.1.mpro"
  open(newfile, "w") do io
    println(io, """
    !----------------------------------------------------------------------
    !   Basis set: cc-pV$(Z)Z-JKFIT + even-tempered diffuse (aug-cc-pV$(Z)Z-JKFIT)
    ! Description: JK Fitting basis for use with aug-cc-pV$(Z)Z
    !        Role: jkfit
    !     Version: 1  (Data from Turbomole 7.3/Molpro + ElemCo.jl)
    !----------------------------------------------------------------------


    spherical
    basis={""")
    ElemCo.output_basis(io, original*"+diffuse")
    println(io, "}")
  end
end


  original = "def2-universal-jkfit"  
  newfile = "aug-def2-universal-jkfit.1.mpro"
  open(newfile, "w") do io
    println(io, """
    !----------------------------------------------------------------------
    !   Basis set: def2-universal-JKFIT + even-tempered diffuse (aug-def2-universal-JKFIT)
    ! Description: Coulomb/Exchange fitting auxiliary basis for def2 basis
    !                    sets
    !        Role: jkfit
    !     Version: 1  (Data from Turbomole 7.3/Molpro + ElemCo.jl)
    !----------------------------------------------------------------------


    spherical
    basis={""")
    ElemCo.output_basis(io, original*"+diffuse")
    println(io, "}")
  end
