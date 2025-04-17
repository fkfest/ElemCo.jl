""" Specify methods available for electron-correlation calculations"""
module ECMethods
using DocStringExtensions
using ..ElemCo.Utils

export ECMethod, method_name, max_full_exc
export has_prefix, set_prefix!, is_unrestricted, set_unrestricted!
export has_suffix, set_suffix!


const ExcLevels = "SDTQP"

const Prefix4Methods = String["EOM-","SVD-","2D-","FRS-","FRT-","Λ","U","R","QV-"]
const Suffix4Methods = String[]

"""
    ECMethod

Description of the electron-correlation method

$(FIELDS)
"""
mutable struct ECMethod
  """theory level: `"MP"`, `"CC"`, `"DC"`."""
  theory::String
  """prefix of the methods, e.g., `"EOM"`, `"U"`, `"R"`, `"2D"`, `"FRS"`, `"FRT"`, `"QV"`."""
  prefix::Vector{String}
  """suffix of the methods."""
  suffix::Vector{String}
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
    theory = ""
    exclevel = [:none for i in 1:length(ExcLevels)]
    pertlevel = 0
    ipos = 1
    # check for prefix
    prefix, ipos = check_specs(Mname, ipos, Prefix4Methods)
    if "EOM" ∈ prefix
      error("EOM methods not implemented!")
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
    elseif substr(Mname, ipos, 4) == "DMRG"
      theory = "DMRG"
      ipos += 4
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
            break
            # error("Excitation level $char not recognized")
          end
          exclevel[iexc] = next_level
        end
      end
      ipos += 1
    end
    suffix, ipos = check_specs(Mname, ipos, Suffix4Methods)
    if ipos <= length(Mname)
      error("Method name not recognized: $mname . Remaining: "*substr(mname,ipos))
    end
    new(theory,prefix,suffix,exclevel,pertlevel)
  end
end

"""
    check_specs(mname::AbstractString, pos, specs::Vector)

  Check if starting from position `pos`, `mname` contains any of `specs`
  and return a list of the matching ones (without dashes for multiple-letter specs!) 
  and the final position after specs.
"""
function check_specs(mname::AbstractString, pos::Int, specs::Vector{String})
  matches = []
  for spec in specs
    if length(mname)-pos+1 >= length(spec)
      if substr(mname, pos, length(spec)) == spec
        if length(spec) > 2 && last(spec) == '-'
          push!(matches, substr(spec,1,length(spec)-1))
        elseif length(spec) > 2 && first(spec) == '-'
          push!(matches, substr(spec,2))
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
    has_prefix(method::ECMethod, spec::AbstractString)

  Return `true` if `method` has prefix `spec`, e.g., `"EOM"`.
"""
function has_prefix(method::ECMethod, spec::AbstractString)
  return spec ∈ method.prefix
end

"""
    set_prefix!(method::ECMethod, spec::AbstractString)

  Set `method` to have prefix `spec`, e.g., `"EOM"`.
"""
function set_prefix!(method::ECMethod, spec::AbstractString)
  if !has_prefix(method, spec)
    push!(method.prefix, spec)
  end
end

"""
    has_suffix(method::ECMethod, spec::AbstractString)

  Return `true` if `method` has suffix `spec`, e.g., `"F12"`.
"""
function has_suffix(method::ECMethod, spec::AbstractString)
  return spec ∈ method.suffix
end

"""
    set_suffix!(method::ECMethod, spec::AbstractString)

  Set `method` to have suffix `spec`, e.g., `"F12"`.
"""
function set_suffix!(method::ECMethod, spec::AbstractString)
  if !has_suffix(method, spec)
    push!(method.suffix, spec)
  end
end

"""
    is_unrestricted(method::ECMethod)

  Return `true` if `method` is unrestricted.
"""
function is_unrestricted(method::ECMethod)
  return has_prefix(method, "U")
end

"""
    set_unrestricted!(method::ECMethod)

  Set `method` to unrestricted.
"""
function set_unrestricted!(method::ECMethod)
  set_prefix!(method, "U")
end

"""
    show(io::IO, method::ECMethod)

  Print `method` to `io`.
"""
function Base.show(io::IO, method::ECMethod)
  print(io, method_name(method))
end

"""
    method_name(method::ECMethod; main::Bool = true, root::Bool = false)

  Return string representation of `method`.
  If `main` is true, return only the main part of the name, i.e., without
  perturbative corrections.
  If `root` is true, return the root name of the method, i.e., without any
  prefixes or suffixes.
"""
function method_name(method::ECMethod; main::Bool = true, root::Bool = false)
  name = ""
  if !root
    for spec in method.prefix
      name *= spec
      if length(spec) > 1
        name *= "-"
      end
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
  if !root
    for spec in method.suffix
      if length(spec) > 1
        name *= "-"
      end
      name *= spec
    end
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