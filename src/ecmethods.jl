""" Specify methods available for electron-correlation calculations"""
module ECMethods
using DocStringExtensions
using ..ElemCo.Utils

export ECMethod, method_name, max_full_exc
export has_spec, set_spec!, is_unrestricted, set_unrestricted!


const ExcLevels = "SDTQP"

const Specs4Methods = ["EOM-","2D-","FRS-","FRT-","Λ","U","R"]

"""
    ECMethod

Description of the electron-correlation method

$(FIELDS)
"""
mutable struct ECMethod
  """unrestricted calculation."""
  unrestricted::Bool
  """theory level: `"MP"`, `"CC"`, `"DC"`."""
  theory::String
  """specification of the methods, e.g., `"EOM"`, `"U"`, `"R"`, `"2D"`, `"FRS"`, `"FRT"`."""
  specs::Vector{String}
  """ excitation level for each class (`exclevel[1]` for singles etc.).
      Possible values: `:none`, `:full`, `:pert`, `:pertiter`. """
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
    Mname = uppercase(mname)
    unrestricted = false
    theory = ""
    exclevel = [:none for i in 1:length(ExcLevels)]
    pertlevel = 0
    ipos = 1
    # check for specs
    specs, ipos = check_specs(Mname, ipos, Specs4Methods)
    if "EOM" ∈ specs
      error("EOM methods not implemented!")
    end
    if 'U' ∈ specs
      unrestricted = true
    end
    # if pure PT: all excitation levels are perturbative, otherwise only the highest
    pure_PT = false
    if substr(Mname, ipos, 2) == "CC"
      theory *= "CC"
      ipos += 2
    elseif substr(Mname, ipos, 2) == "DC"
      if substr(Mname, ipos, 5) == "DC-CC"
        theory *= "DC-CC"
        ipos += 5
      else
        theory *= "DC"
        ipos += 2
      end
    elseif substr(Mname, ipos, 2) == "MP"
      theory = "MP"
      ipos += 2
      pure_PT = true
    else
      error("Theory not recognized in "*mname*": "*substr(Mname,ipos,2))
    end
    # loop over remaining letters to get excitation levels
    # currently case-insensitive, can change later...
    next_level = :full
    for char in substr(Mname, ipos)
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
    new(unrestricted,theory,specs,exclevel,pertlevel)
  end
end

"""
    check_specs(mname::AbstractString, pos, specs::Vector)

  Check if starting from position `pos`, `mname` contains any of `specs`
  and return a list of the matching ones (without dashes for multiple-letter specs!) 
  and the final position after specs.
"""
function check_specs(mname::AbstractString, pos, specs::Vector)
  matches = []
  for spec in specs
    if length(mname)-pos+1 >= length(spec)
      if substr(mname, pos, length(spec)) == spec
        if length(spec) > 2 && last(spec) == '-'
          push!(matches, substr(spec,1,length(spec)-1))
        else
          push!(matches, spec)
        end
        pos += length(spec)
      end
    end
  end
  return matches, pos
end

"""
    has_spec(method::ECMethod, spec::AbstractString)

  Return `true` if `method` has specification `spec`, e.g., `"EOM"`.
"""
function has_spec(method::ECMethod, spec::AbstractString)
  return spec ∈ method.specs
end

"""
    set_spec!(method::ECMethod, spec::AbstractString)

  Set `method` to have specification `spec`, e.g., `"EOM"`.
"""
function set_spec!(method::ECMethod, spec::AbstractString)
  if !has_spec(method, spec)
    push!(method.specs, spec)
  end
end

"""
    is_unrestricted(method::ECMethod)

  Return `true` if `method` is unrestricted.
"""
function is_unrestricted(method::ECMethod)
  return has_spec(method, "U")
end

"""
    set_unrestricted!(method::ECMethod)

  Set `method` to unrestricted.
"""
function set_unrestricted!(method::ECMethod)
  set_spec!(method, "U")
end

"""
    show(io::IO, method::ECMethod)

  Print `method` to `io`.
"""
function Base.show(io::IO, method::ECMethod)
  print(io, method_name(method))
end

"""
    method_name(method::ECMethod, main::Bool = true)

  Return string representation of `method`.
  If `main` is true, return only the main part of the name, i.e., without
  perturbative corrections.
"""
function method_name(method::ECMethod, main::Bool = true)
  name = ""
  for spec in method.specs
    name *= spec
    if length(spec) > 1
      name *= "-"
    end
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