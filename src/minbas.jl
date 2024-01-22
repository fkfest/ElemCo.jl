

""" ANO-R0 minimal basis set. """
const NSHELL4L_ANO_R0 = [ 
  fill([1], 2)...,      # H, HE 
  fill([2,1], 8)...,    # LI, BE, B, C, N, O, NE 
  fill([3,1], 2)...,    # NA, MG
  fill([3,2], 6)...,    # AL, SI, P, S, CL, AR
  fill([4,2], 2)...,    # K, CA
  fill([4,2,1], 10)..., # SC, TI, V, CR, MN, FE, CO, NI, CU, ZN
  fill([4,3,1], 6)...,  # GA, GE, AS, SE, BR, KR
  fill([5,3,1], 2)...,  # RB, SR
  fill([5,3,2], 10)..., # Y, ZR, NB, MO, TC, RU, RH, PD, AG, CD
  fill([5,4,2], 6)...,  # IN, SN, SB, TE, I, XE
  fill([6,4,2], 2)...,  # CS, BA
  [6,4,3],              # LA
  [6,4,3,1],            # CE
  fill([6,4,2,1], 5)...,# PR, ND, PM, SM, EU 
  [6,4,3,1],            # GD
  fill([6,4,2,1], 6)...,# TB, DY, HO, ER, TM, YB
  fill([6,4,3,1], 10)..., # LU, HF, TA, W, RE, OS, IR, PT, AU, HG
  fill([6,5,3,1], 6)... # TL, PB, BI, PO, AT, RN 
]

""" ANO-RCC-MB minimal basis set. """
const NSHELL4L_ANO_RCC_MB = [ 
  fill([1], 2)...,      # H, HE
  fill([2,1], 8)...,    # LI, BE, B, C, N, O, F, NE
  fill([3,2], 8)...,    # NA, MG, AL, SI, P, S, CL, AR
  fill([4,3], 2)...,    # K, CA
  fill([4,3,1], 16)..., # SC, TI, V, CR, MN, FE, CO, NI, CU, ZN, GA, GE, AS, SE, BR, KR
  fill([5,4,1], 2)...,  # RB, SR
  fill([5,4,2], 16)..., # Y, ZR, NB, MO, TC, RU, RH, PD, AG, CD, IN, SN, SB, TE, I, XE
  fill([6,5,2], 2)...,  # CS, BA
  fill([6,5,3,1], 30)..., # LA, CE, PR, ND, PM, SM, EU, GD, TB, DY, HO, ER, TM, YB, LU, HF, TA, W, RE, OS, IR, PT, AU, HG, TL, PB, BI, PO, AT, RN 
  fill([7,6,3,1], 2)...,# FR, RA 
  fill([7,6,4,2], 8)... # AC, TH, PA, U, NP, PU, AM, CM 
]

""" Number of shells for each angular momentum in the minimal basis set """
const NSHELL4L_MINBAS = Dict(
  "ANO-RCC-MB" => NSHELL4L_ANO_RCC_MB,
  "ANO-R0"     => NSHELL4L_ANO_R0
)

""" 
    nshell4l_minbas(nnum, basis::String)

  Return the number of shells for each angular momentum in the minimal basis set. 
"""
function nshell4l_minbas(nnum, basis::String)
  if haskey(NSHELL4L_MINBAS, basis)
    return NSHELL4L_MINBAS[basis][nnum]
  else
    error("Minbas dimenstions not available for basis $basis")
  end
end