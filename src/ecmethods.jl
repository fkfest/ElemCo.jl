""" Specify methods available for electron-correlation calculations"""
module ECMethods
using DocStringExtensions

export ECMethod, method_name, max_full_exc


const ExcLevels = "SDTQP"

"""
    ECMethod

Description of the electron-correlation method

$(FIELDS)
"""
mutable struct ECMethod
  """unrestricted calculation."""
  unrestricted::Bool
  """theory level: MP, CC, DC."""
  theory::String
  """ excitation level for each class (exclevel[1] for singles etc.).
      Possible values: :none, :full, :pert, :pertiter. """
  exclevel::Array{Symbol,1}
  """ perturbation theory level (relevant for MP methods)."""
  pertlevel::Int
  """
      ECMethod(mname::AbstractString)" 

    Parse method name `mname` and return `ECMethod` object.
  """
  function ECMethod(mname::AbstractString)
    if isempty(mname)
      error("Empty method name!")
    end
    unrestricted = false
    theory = ""
    exclevel = [:none for i in 1:length(ExcLevels)]
    pertlevel = 0
    ipos = 1
    if uppercase(mname[ipos:ipos+2]) == "EOM"
      error("EOM methods not implemented!")
      ipos += 3
      if mname[ipos] == '-'
        ipos += 1
      end
    end
    if uppercase(mname[ipos:ipos+1]) == "2D"
      theory = "2D-"
      ipos += 3
    elseif uppercase(mname[ipos:ipos+2]) == "FRS"
      theory = "FRS-"
      ipos += 4
    elseif uppercase(mname[ipos:ipos+2]) == "FRT"
      theory = "FRT-"
      ipos += 4
    end
    if uppercase(mname[ipos]) == 'U'
      unrestricted = true
      ipos += 1
    end
    # if pure PT: all excitation levels are perturbative, otherwise only the highest
    pure_PT = false
    if uppercase(mname[ipos:ipos+1]) == "CC"
      theory *= "CC"
      ipos += 2
    elseif uppercase(mname[ipos:ipos+1]) == "DC"
      if length(mname)-ipos >= 4 && uppercase(mname[ipos:ipos+4]) == "DC-CC"
        theory *= "DC-CC"
        ipos += 5
      else
        theory *= "DC"
        ipos += 2
      end
    elseif uppercase(mname[ipos:ipos+1]) == "MP"
      theory = "MP"
      ipos += 2
      pure_PT = true
    else
      error("Theory not recognized in "*mname*": "*uppercase(mname[ipos:ipos+1]))
    end
    # loop over remaining letters to get excitation levels
    # currently case-insensitive, can change later...
    next_level = :full
    for char in uppercase(mname[ipos:end])
      if isnumeric(char)
        # perturbation theory
        level = parse(Int,char)
        if pure_PT
          pertlevel = level
          # 2n+1 rule for perturbation theory
          level_exc = level ÷ 2 + 1
          lower_level = :pert
          exclevel[level_exc] = :pert
        else
          # for CCn methods n = excitation level
          level_exc = level
          lower_level = :full
          exclevel[level_exc] = :pertiter
        end
        for i in 1:level_exc-1
          if exclevel[i] == :none
            exclevel[i] = lower_level
          end
        end
      else  
        if char == '('
          next_level = :pert
        elseif char == ')'
          next_level = :full
        else
          iexc = findfirst(char,ExcLevels)
          if isnothing(iexc)
            error("Excitation level $char not recognized")
          end
          exclevel[iexc] = next_level
        end
      end
    end
    new(unrestricted,theory,exclevel,pertlevel)
  end
end

"""
    method_name(method::ECMethod, main::Bool = true)

  Return string representation of `method`.
  If `main` is true, return only the main part of the name, i.e., without
  perturbative corrections.
"""
function method_name(method::ECMethod, main::Bool = true)
  name = ""
  if method.unrestricted
    name *= "U"
  end
  name *= method.theory
  if method.theory == "MP"
    name *= string(method.pertlevel)
  else
    level_str = ""
    for (i,ex) in enumerate(ExcLevels)
      if method.exclevel[i] == :pert
        if !main
          level_str *= "($ex)"
        end
      elseif method.exclevel[i] == :full
        level_str *= ex
      elseif method.exclevel[i] == :none
        continue
      elseif method.exclevel[i] == :pertiter
        level_str = string(i)
      else
        error("Excitation level not recognized!")
      end
    end
    name *= level_str
  end
  return name
end

"""
    max_full_exc(method::ECMethod)

  Return the highest full excitation level of the `method`.
"""
function max_full_exc(method::ECMethod)
  for (i,ex) in enumerate(reverse(method.exclevel))
    if ex == :full
      return length(method.exclevel)-i+1
    end
  end
  return 0
end

end #module