""" Specify ecmethods """
module ECMethods

export ECMethod, ExcType, NoExc, FullExc, PertExc

const ExcLevels = "SDTQP"

@enum ExcType NoExc FullExc PertExc
"""
Description of the electron-correlation method
"""
struct ECMethod
  unrestricted::Bool
  """theory level: MP, CC, DC"""
  theory::String
  """ excitation level for each class (exclevel[1] for singles etc.)"""
  exclevel::Array{ExcType}

  function ECMethod(mname::AbstractString)
    if isempty(mname)
      error("Empty method name!")
    end
    unrestricted = false
    theory = ""
    exclevel = [NoExc for i in 1:length(ExcLevels)]
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
    if uppercase(mname[ipos:ipos+1]) == "CC"
      theory = "CC"
      ipos += 2
    elseif uppercase(mname[ipos:ipos+1]) == "DC"
      theory = "DC"
      ipos += 2
    elseif uppercase(mname[ipos:ipos+1]) == "MP"
      theory = "MP"
      ipos += 2
    else
      error("Theory not recognized in "*mname*": "*uppercase(mname[ipos:ipos+1]))
    end
    # loop over remaining letters to get excitation levels
    # currently case-insensitive, can change later...
    next_level = FullExc
    for char in uppercase(mname[ipos:end])
      if char == '2'
        if exclevel[1] == NoExc
          exclevel[1] = PertExc
        end
        exclevel[2] = PertExc
      else 
        if char == '('
          next_level = PertExc
        elseif char == ')'
          next_level = FullExc
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