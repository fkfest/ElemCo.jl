""" Specify methods available for electron-correlation calculations"""
module ECMethods
using DocStringExtensions

export ECMethod

const ExcLevels = "SDTQP"

"""
Description of the electron-correlation method
"""
struct ECMethod
  """unrestricted calculation."""
  unrestricted::Bool
  """theory level: MP, CC, DC."""
  theory::String
  """ excitation level for each class (exclevel[1] for singles etc.).
      Possible values: :none, :full, :pert, :pertiter. """
  exclevel::Array{Symbol,1}
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
    ipos = 1
    if uppercase(mname[ipos:ipos+2]) == "EOM"
      error("EOM methods not implemented!")
      ipos += 3
      if mname[ipos] == '-'
        ipos += 1
      end
    end
    if uppercase(mname[ipos]) == 'U'
      unrestricted = true
      ipos += 1
    end
    # if pure PT: all excitation levels are perturbative, otherwise only the highest
    pure_PT = false
    if uppercase(mname[ipos:ipos+1]) == "CC"
      theory = "CC"
      ipos += 2
    elseif uppercase(mname[ipos:ipos+1]) == "DC"
      if length(mname)-ipos >= 4 && uppercase(mname[ipos:ipos+4]) == "DC-CC"
        theory = "DC-CC"
        ipos += 5
      else
        theory = "DC"
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
          lower_level = :pert
          exclevel[level] = :pert
        else
          lower_level = :full
          exclevel[level] = :pertiter
        end
        for i in 1:level-1
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
    new(unrestricted,theory,exclevel)
  end
end




end #module