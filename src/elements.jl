
"""
  Elements with corresponding atomic numbers, 
  atomic masses, name, electron configuration, large core, small core (w/o semi-core).
"""
const ELEMENTS = Dict(
  "H"  =>  [1,  1.00784, "Hydrogen"   , "1s^1"                , ""            , ""] ,
  "HE" =>  [2,  4.00260, "Helium"     , "1s^2"                , ""            , ""],
  "LI" =>  [3,  6.938  , "Lithium"    , "[HE]2s^1"            , "[HE]"        , "[HE]"],
  "BE" =>  [4,  9.01218, "Beryllium"  , "[HE]2s^2"            , "[HE]"        , "[HE]"],
  "B"  =>  [5, 10.806  , "Boron"      , "[HE]2s^2 2p^1"       , "[HE]"        , "[HE]"],
  "C"  =>  [6, 12.0096 , "Carbon"     , "[HE]2s^2 2p^2"       , "[HE]"        , "[HE]"],
  "N"  =>  [7, 14.00643, "Nitrogen"   , "[HE]2s^2 2p^3"       , "[HE]"        , "[HE]"],
  "O"  =>  [8, 15.99903, "Oxygen"     , "[HE]2s^2 2p^4"       , "[HE]"        , "[HE]"],
  "F"  =>  [9, 18.99840, "Fluorine"   , "[HE]2s^2 2p^5"       , "[HE]"        , "[HE]"],
  "NE" => [10, 20.1797 , "Neon"       , "[HE]2s^2 2p^6"       , "[HE]"        , "[HE]"],
  "NA" => [11, 22.98977, "Sodium"     , "[NE]3s^1"            , "[NE]"        , "[NE]"],
  "MG" => [12, 24.304  , "Magnesium"  , "[NE]3s^2"            , "[NE]"        , "[NE]"],
  "AL" => [13, 26.9815 , "Aluminum"   , "[NE]3s^2 3p^1"       , "[NE]"        , "[NE]"],
  "SI" => [14, 28.0855 , "Silicon"    , "[NE]3s^2 3p^2"       , "[NE]"        , "[NE]"],
  "P"  => [15, 30.97376, "Phosphorus" , "[NE]3s^2 3p^3"       , "[NE]"        , "[NE]"],
  "S"  => [16, 32.065  , "Sulfur"     , "[NE]3s^2 3p^4"       , "[NE]"        , "[NE]"],
  "CL" => [17, 35.453  , "Chlorine"   , "[NE]3s^2 3p^5"       , "[NE]"        , "[NE]"],
  "AR" => [18, 39.948  , "Argon"      , "[NE]3s^2 3p^6"       , "[NE]"        , "[NE]"],
  "K"  => [19, 39.0983 , "Potassium"  , "[AR]4s^1"            , "[AR]"        , "[AR]"],
  "CA" => [20, 40.078  , "Calcium"    , "[AR]4s^2"            , "[AR]"        , "[AR]"],
  "SC" => [21, 44.9559 , "Scandium"   , "[AR]3d^1 4s^2"       , "[AR]"        , "[NE]"],
  "TI" => [22, 47.867  , "Titanium"   , "[AR]3d^2 4s^2"       , "[AR]"        , "[NE]"],
  "V"  => [23, 50.9415 , "Vanadium"   , "[AR]3d^3 4s^2"       , "[AR]"        , "[NE]"],
  "CR" => [24, 51.9961 , "Chromium"   , "[AR]3d^5 4s^1"       , "[AR]"        , "[NE]"],
  "MN" => [25, 54.9380 , "Manganese"  , "[AR]3d^5 4s^2"       , "[AR]"        , "[NE]"],
  "FE" => [26, 55.845  , "Iron"       , "[AR]3d^6 4s^2"       , "[AR]"        , "[NE]"],
  "CO" => [27, 58.9332 , "Cobalt"     , "[AR]3d^7 4s^1"       , "[AR]"        , "[NE]"],
  "NI" => [28, 58.6934 , "Nickel"     , "[AR]3d^8 4s^2"       , "[AR]"        , "[NE]"],
  "CU" => [29, 63.546  , "Copper"     , "[AR]3d^10 4s^1"      , "[AR]3d^10"   , "[NE]"],
  "ZN" => [30, 65.38   , "Zinc"       , "[AR]3d^10 4s^2"      , "[AR]3d^10"   , "[AR]3d^10"],
  "GA" => [31, 69.723  , "Gallium"    , "[AR]3d^10 4s^2 4p^1" , "[AR]3d^10"   , "[AR]3d^10"],
  "GE" => [32, 72.64   , "Germanium"  , "[AR]3d^10 4s^2 4p^2" , "[AR]3d^10"   , "[AR]3d^10"],
  "AS" => [33, 74.9216 , "Arsenic"    , "[AR]3d^10 4s^2 4p^3" , "[AR]3d^10"   , "[AR]3d^10"],
  "SE" => [34, 78.96   , "Selenium"   , "[AR]3d^10 4s^2 4p^4" , "[AR]3d^10"   , "[AR]3d^10"],
  "BR" => [35, 79.904  , "Bromine"    , "[AR]3d^10 4s^2 4p^5" , "[AR]3d^10"   , "[AR]3d^10"],
  "KR" => [36, 83.80   , "Krypton"    , "[AR]3d^10 4s^2 4p^6" , "[AR]3d^10"   , "[AR]3d^10"],
  "RB" => [37, 85.4678 , "Rubidium"   , "[KR]5s^1"            , "[KR]"        , "[KR]"],
  "SR" => [38, 87.62   , "Strontium"  , "[KR]5s^2"            , "[KR]"        , "[KR]"],
  "Y"  => [39, 88.9059 , "Yttrium"    , "[KR]4d^1 5s^2"       , "[KR]"        , "[AR]"],
  "ZR" => [40, 91.224  , "Zirconium"  , "[KR]4d^2 5s^2"       , "[KR]"        , "[AR]"],
  "NB" => [41, 92.9064 , "Niobium"    , "[KR]4d^4 5s^1"       , "[KR]"        , "[AR]"],
  "MO" => [42, 95.94   , "Molybdenum" , "[KR]4d^5 5s^1"       , "[KR]"        , "[AR]"],
  "TC" => [43, 98.0    , "Technetium" , "[KR]4d^5 5s^2"       , "[KR]"        , "[AR]"],
  "RU" => [44, 101.07  , "Ruthenium"  , "[KR]4d^7 5s^1"       , "[KR]"        , "[AR]"],
  "RH" => [45, 102.9055, "Rhodium"    , "[KR]4d^8 5s^1"       , "[KR]"        , "[AR]"],
  "PD" => [46, 106.42  , "Palladium"  , "[KR]4d^10"           , "[KR]"        , "[AR]"],
  "AG" => [47, 107.8682, "Silver"     , "[KR]4d^10 5s^1"      , "[PD]"        , "[AR]"],
  "CD" => [48, 112.411 , "Cadmium"    , "[KR]4d^10 5s^2"      , "[PD]"        , "[PD]"],
  "IN" => [49, 114.818 , "Indium"     , "[KR]4d^10 5s^2 5p^1" , "[PD]"        , "[PD]"],
  "SN" => [50, 118.710 , "Tin"        , "[KR]4d^10 5s^2 5p^2" , "[PD]"        , "[PD]"],
  "SB" => [51, 121.760 , "Antimony"   , "[KR]4d^10 5s^2 5p^3" , "[PD]"        , "[PD]"],
  "TE" => [52, 127.60  , "Tellurium"  , "[KR]4d^10 5s^2 5p^4" , "[PD]"        , "[PD]"],
  "I"  => [53, 126.9045, "Iodine"     , "[KR]4d^10 5s^2 5p^5" , "[PD]"        , "[PD]"],
  "XE" => [54, 131.29  , "Xenon"      , "[KR]4d^10 5s^2 5p^6" , "[PD]"        , "[PD]"],
  "CS" => [55, 132.9054, "Caesium"    , "[XE]6s^1"            , "[XE]"        , "[XE]"],
  "BA" => [56, 137.327 , "Barium"     , "[XE]6s^2"            , "[XE]"        , "[XE]"],
  "LA" => [57, 138.9055, "Lanthanum"  , "[XE]5d^1 6s^2"       , "[XE]"        , "[KR]"],
  "CE" => [58, 140.116 , "Cerium"     , "[XE]4f^1 5d^1 6s^2"  , "[XE]"        , "[KR]"],
  "PR" => [59, 140.9077,"Praseodymium", "[XE]4f^3 6s^2"       , "[XE]"        , "[KR]"],
  "ND" => [60, 144.24  , "Neodymium"  , "[XE]4f^4 6s^2"       , "[XE]"        , "[KR]"],
  "PM" => [61, 145.0   , "Promethium" , "[XE]4f^5 6s^2"       , "[XE]"        , "[KR]"],
  "SM" => [62, 150.36  , "Samarium"   , "[XE]4f^6 6s^2"       , "[XE]"        , "[KR]"],
  "EU" => [63, 151.964 , "Europium"   , "[XE]4f^7 6s^2"       , "[XE]"        , "[KR]"],
  "GD" => [64, 157.25  , "Gadolinium" , "[XE]4f^7 5d^1 6s^2"  , "[XE]"        , "[KR]"],
  "TB" => [65, 158.9254, "Terbium"    , "[XE]4f^9 6s^2"       , "[XE]"        , "[KR]"],
  "DY" => [66, 162.50  , "Dysprosium" , "[XE]4f^10 6s^2"      , "[XE]"        , "[KR]"],
  "HO" => [67, 164.9304, "Holmium"    , "[XE]4f^11 6s^2"      , "[XE]"        , "[KR]"],
  "ER" => [68, 167.26  , "Erbium"     , "[XE]4f^12 6s^2"      , "[XE]"        , "[KR]"],
  "TM" => [69, 168.9342, "Thulium"    , "[XE]4f^13 6s^2"      , "[XE]"        , "[KR]"],
  "YB" => [70, 173.04  , "Ytterbium"  , "[XE]4f^14 6s^2"      , "[XE]4f^14"   , "[KR]"],
  "LU" => [71, 174.967 , "Lutetium"   , "[XE]4f^14 5d^1 6s^2" , "[XE]4f^14"   , "[KR]"],
  "HF" => [72, 178.49  , "Hafnium"    , "[XE]4f^14 5d^2 6s^2" , "[XE]4f^14"   , "[KR]"],
  "TA" => [73, 180.9479, "Tantalum"   , "[XE]4f^14 5d^3 6s^2" , "[XE]4f^14"   , "[KR]"],
  "W"  => [74, 183.84  , "Tungsten"   , "[XE]4f^14 5d^4 6s^2" , "[XE]4f^14"   , "[KR]"],
  "RE" => [75, 186.207 , "Rhenium"    , "[XE]4f^14 5d^5 6s^2" , "[XE]4f^14"   , "[KR]"],
  "OS" => [76, 190.23  , "Osmium"     , "[XE]4f^14 5d^6 6s^2" , "[XE]4f^14"   , "[KR]"],
  "IR" => [77, 192.217 , "Iridium"    , "[XE]4f^14 5d^7 6s^2" , "[XE]4f^14"   , "[KR]"],
  "PT" => [78, 195.078 , "Platinum"   , "[XE]4f^14 5d^9 6s^1" , "[XE]4f^14"   , "[KR]"],
  "AU" => [79, 196.9665, "Gold"       , "[XE]4f^14 5d^10 6s^1", "[XE]4f^14 5d^10", "[KR]"],
  "HG" => [80, 200.59  , "Mercury"    , "[XE]4f^14 5d^10 6s^2", "[XE]4f^14 5d^10", "[KR]"],
  "TL" => [81, 204.3833, "Thallium"   , "[XE]4f^14 5d^10 6s^2 6p^1", "[XE]4f^14 5d^10", "[KR]"],
  "PB" => [82, 207.2   , "Lead"       , "[XE]4f^14 5d^10 6s^2 6p^2", "[XE]4f^14 5d^10", "[KR]"],
  "BI" => [83, 208.9804, "Bismuth"    , "[XE]4f^14 5d^10 6s^2 6p^3", "[XE]4f^14 5d^10", "[KR]"],
  "PO" => [84, 209.0   , "Polonium"   , "[XE]4f^14 5d^10 6s^2 6p^4", "[XE]4f^14 5d^10", "[KR]"],
  "AT" => [85, 210.0   , "Astatine"   , "[XE]4f^14 5d^10 6s^2 6p^5", "[XE]4f^14 5d^10", "[KR]"],
  "RN" => [86, 222.0   , "Radon"      , "[XE]4f^14 5d^10 6s^2 6p^6", "[XE]4f^14 5d^10", "[KR]"],
  "FR" => [87, 223.0   , "Francium"   , "[RN]7s^1"            , "[RN]"        , "[RN]"],
  "RA" => [88, 226.0   , "Radium"     , "[RN]7s^2"            , "[RN]"        , "[RN]"],
  "AC" => [89, 227.0   , "Actinium"   , "[RN]6d^1 7s^2"       , "[RN]"        , "[XE]"],
  "TH" => [90, 232.0381, "Thorium"    , "[RN]6d^2 7s^2"       , "[RN]"        , "[XE]"],
  "PA" => [91, 231.0359,"Protactinium", "[RN]5f^2 6d^1 7s^2"  , "[RN]"        , "[XE]"],
  "U"  => [92, 238.0289, "Uranium"    , "[RN]5f^3 6d^1 7s^2"  , "[RN]"        , "[XE]"],
  "NP" => [93, 237.0   , "Neptunium"  , "[RN]5f^4 6d^1 7s^2"  , "[RN]"        , "[XE]"],
  "PU" => [94, 244.0   , "Plutonium"  , "[RN]5f^6 7s^2"       , "[RN]"        , "[XE]"],
  "AM" => [95, 243.0   , "Americium"  , "[RN]5f^7 7s^2"       , "[RN]"        , "[XE]"],
  "CM" => [96, 247.0   , "Curium"     , "[RN]5f^7 6d^1 7s^2"  , "[RN]"        , "[XE]"],
  "BK" => [97, 247.0   , "Berkelium"  , "[RN]5f^9 7s^2"       , "[RN]"        , "[XE]"],
  "CF" => [98, 251.0   , "Californium", "[RN]5f^10 7s^2"      , "[RN]"        , "[XE]"],
  "ES" => [99, 252.0   , "Einsteinium", "[RN]5f^11 7s^2"      , "[RN]"        , "[XE]"],
  "FM" =>[100, 257.0   , "Fermium"    , "[RN]5f^12 7s^2"      , "[RN]"        , "[XE]"],
  "MD" =>[101, 258.0   , "Mendelevium", "[RN]5f^13 7s^2"      , "[RN]"        , "[XE]"],
  "NO" =>[102, 259.0   , "Nobelium"   , "[RN]5f^14 7s^2"      , "[RN]5f^14"   , "[XE]"],
  "LR" =>[103, 262.0   , "Lawrencium" , "[RN]5f^14 7s^2 7p^1" , "[RN]5f^14"   , "[XE]"],
  "RF" =>[104, 261.0   ,"Rutherfordium", "[RN]5f^14 6d^2 7s^2", "[RN]5f^14"   , "[XE]"],
  "DB" =>[105, 262.0   , "Dubnium"    , "[RN]5f^14 6d^3 7s^2" , "[RN]5f^14"   , "[XE]"],
  "SG" =>[106, 266.0   , "Seaborgium" , "[RN]5f^14 6d^4 7s^2" , "[RN]5f^14"   , "[XE]"],
  "BH" =>[107, 264.0   , "Bohrium"    , "[RN]5f^14 6d^5 7s^2" , "[RN]5f^14"   , "[XE]"],
  "HS" =>[108, 277.0   , "Hassium"    , "[RN]5f^14 6d^6 7s^2" , "[RN]5f^14"   , "[XE]"],
  "MT" =>[109, 268.0   , "Meitnerium" , "[RN]5f^14 6d^7 7s^2" , "[RN]5f^14"   , "[XE]"],
  "DS" =>[110, 281.0   ,"Darmstadtium", "[RN]5f^14 6d^9 7s^1" , "[RN]5f^14"   , "[XE]"],
  "RG" =>[111, 272.0   , "Roentgenium", "[RN]5f^14 6d^10 7s^1", "[RN]5f^14 6d^10", "[XE]"],
  "CN" =>[112, 285.0   , "Copernicium", "[RN]5f^14 6d^10 7s^2", "[RN]5f^14 6d^10", "[XE]"],
  "NH" =>[113, 284.0   , "Nihonium"   , "[RN]5f^14 6d^10 7s^2 7p^1", "[RN]5f^14 6d^10", "[XE]"],
  "FL" =>[114, 289.0   , "Flerovium"  , "[RN]5f^14 6d^10 7s^2 7p^2", "[RN]5f^14 6d^10", "[XE]"],
  "MC" =>[115, 288.0   , "Moscovium"  , "[RN]5f^14 6d^10 7s^2 7p^3", "[RN]5f^14 6d^10", "[XE]"],
  "LV" =>[116, 293.0   , "Livermorium", "[RN]5f^14 6d^10 7s^2 7p^4", "[RN]5f^14 6d^10", "[XE]"],
  "TS" =>[117, 294.0   , "Tennessine" , "[RN]5f^14 6d^10 7s^2 7p^5", "[RN]5f^14 6d^10", "[XE]"],
  "OG" =>[118, 294.0   , "Oganesson"  , "[RN]5f^14 6d^10 7s^2 7p^6", "[RN]5f^14 6d^10", "[XE]"]
  )

"""
    nuclear_charge_of_center(elem::AbstractString)

  Return the nuclear charge of the element.
"""
function nuclear_charge_of_center(elem::AbstractString)
  return get(ELEMENTS, uppercase(elem), [0])[1]
end

const SUBSHELLS_NAMES = "spdfghi"

const SUBSHELL2L = Dict('s'=>0,'p'=>1,'d'=>2,'f'=>3,'g'=>4,'h'=>5,'i'=>6)

"""
    n_orbitals_in_subshell(shell::Char)

  Return the number of orbitals in the subshell.
"""
n_orbitals_in_subshell(shell::Char) = 2*SUBSHELL2L[shell]+1
n_orbitals_in_subshell(lnum::Int) = 2*lnum+1

""" 
  Occupation of the subshell with quantum numbers n and l.
"""
struct SubShell
  n::Int
  l::Int
  nel::Int
end

""" 
    parse_electron_configuration(e::AbstractString)

  Parse the electron configuration string and return the number of electrons in each subshell.
  e.g. "[He] 2s^2 2p^6 3s^2 3p^6" -> [SubShell(1,0,2), SubShell(2,0,2), SubShell(2,1,6), SubShell(3,0,2), SubShell(3,1,6)] 
"""
function parse_electron_configuration(e::AbstractString)
  subshells = SubShell[]
  if e == ""
    return subshells
  end
  econf = e
  while econf[1] == '['
    el_core = econf[2:findfirst(']', econf)-1]
    econf = replace(econf, "[$el_core]" => ELEMENTS[el_core][4]*" ")
  end
  shells = split(econf)
  for sh in shells
    level = parse(Int, sh[1])
    lnum = SUBSHELL2L[sh[2]]
    nel = parse(Int, sh[4:end])
    push!(subshells, SubShell(level, lnum, nel))
  end
  return subshells
end

""" 
    electron_distribution(elem::AbstractString, nsh4l::Vector{Int})

  Distribute electrons among first atomic orbitals in nsh4l[1]s nsh4l[2]p nsh4l[3]d nsh4l[4]f... order
  considering the Hund's rule and electron configuration of the atom.
  Average occupations to account for the spin degeneracy and hybridization.
"""
function electron_distribution(elem::AbstractString, nsh4l::Vector{Int})
  eldist = Float64[]
  n = nuclear_charge_of_center(elem)
  if n == 0
    return eldist
  end
  subshells = parse_electron_configuration(ELEMENTS[elem][4])
  maxl = maximum([sh.l for sh in subshells])
  if maxl > length(nsh4l)
    error("too many shells in the electron configuration of $elem")
  end
  coresubshells = parse_electron_configuration(ELEMENTS[elem][5])
  val = [(sh.nel, n_orbitals_in_subshell(sh.l)) for sh in subshells if sh ∉ coresubshells]
  nvalel = sum(first.(val))
  nvalorb = sum(last.(val))
  el4l = [zeros(Float64,n_orbitals_in_subshell(l-1)*nsh4l[l]) for l = 1:length(nsh4l)]
  ish4l = zeros(Int,length(nsh4l))
  for sh in subshells
    if sh.nel > n_orbitals_in_subshell(sh.l)*2
      error("too many electrons in the subshell $(sh.n)$(SUBSHELLS_NAMES[sh.l+1])")
    end
    norb = n_orbitals_in_subshell(sh.l)
    ist = ish4l[sh.l+1]+1
    ien = ish4l[sh.l+1]+norb
    if ien > length(el4l[sh.l+1])
      error("too many shells in the electron configuration of $elem")
    end
    if sh ∈ coresubshells
      el4l[sh.l+1][ist:ien] = fill(2.0, norb)
    else
      el4l[sh.l+1][ist:ien] = fill(nvalel/nvalorb, norb)
    end
    ish4l[sh.l+1] += norb
  end
  eldist = vcat(eldist, el4l...)
end
