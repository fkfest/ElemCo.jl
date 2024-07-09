"""
Various constants for the package.

Individual constants are not exported. Use `Constants.<name>` to access them.
"""
module Constants

# Major (exact) constants
""" speed of light in vacuum [``m~s^{-1}``]"""
const C = 299792458
""" Boltzmann constant [``J~K^{-1}``]"""
const KB = 1.380649e-23
""" Planck constant [``J~s``]"""
const H = 6.62607015e-34
""" elementary charge [``C``]"""
const E = 1.602176634e-19
""" Avogadro constant [``mol^{-1}``]"""
const NA = 6.02214076e23

# Measured and combined constants
""" reduced Planck constant [``J~s``]"""
const HBAR = H/(2*pi) # 1.0545718176e-34
""" vacuum magnetic permeability [``N~A^{-2}``]"""
const MU0 = 1.25663706212e-6
""" vacuum electric permittivity [``F~m^{-1}``]"""
const EPS0 = 1/(MU0*C^2) # 8.8541878128e-12
""" electron mass [``kg``]"""
const ME = 9.1093837015e-31
""" proton mass [``kg``]"""
const MP = 1.67262192369e-27
""" neutron mass [``kg``]"""
const MN = 1.67492749804e-27
""" Bohr radius [``m``]"""
const A0 = EPS0*H^2/(pi*ME*E^2) # 5.291772109e-11
""" Hartree energy [``J``]"""
const HARTREE = HBAR^2/(ME*A0^2) # 4.359744722e-18
""" atomic mass unit [``kg``]"""
const AMU = 1.66053906660e-27

# Other constants
""" kcal/mol [``J~mol^{-1}``]"""
const KCAL = 4184
""" Debye [``C~m``]"""
const DEBYE = 1e-21/C # 3.33564095198152e-30


# Transformations between units 
# length
""" Bohr to meter [``m~au^{-1}``]"""
const BOHR2METER = A0
""" Bohr to angstrom [``\\AA~au^{-1}``]"""
const BOHR2ANGSTROM = A0*1e10 # 0.5291772109
# energy
""" Hartree to eV [``eV~E_h^{-1}``]"""
const HARTREE2EV = HARTREE/E # 27.211386246
""" Hartree to kcal/mol [``kcal~mol^{-1}~E_h^{-1}``]"""
const HARTREE2KCAL = HARTREE*NA/KCAL # 627.509474
""" Hartree to kJ/mol [``kJ~mol^{-1}~E_h^{-1}``]"""
const HARTREE2KJ = HARTREE*NA/1000 # 2625.499639
""" Hartree to cm^{-1} [``cm^{-1}~E_h^{-1}``]"""
const HARTREE2CM = HARTREE/(H*C*100) # 219474.63
""" Hartree to Kelvin [``K~E_h^{-1}``]"""
const HARTREE2K = HARTREE/KB # 315775.025
# dipole moment
""" au to Debye [``D~au^{-1}``]"""
const AU2DEBYE = E*A0/DEBYE # 2.541746
# mass
""" atomic mass unit to kg [``kg~u^{-1}``]"""
const AMU2KG = AMU

end #module