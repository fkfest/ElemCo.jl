# ============================================================================
# Non-standard field extensions (violate the TREXIO standard)
# ============================================================================

## Basis set extensions
const TREXIO_BASIS_EXT_FIELDS = [
    TrexioField("basis", "name", String, SCALAR, "name of the basis set", violator=true),
]

## One-electron AO integral extensions (ao_1e_int group) - non-standard, violator
const TREXIO_AO_1E_INT_EXT_FIELDS = [
    TrexioField("ao_1e_int", "fock", Float64, ["ao.num", "ao.num"], "AO-basis Fock matrix (restricted/closed-shell)", violator=true),
    TrexioField("ao_1e_int", "fock_up", Float64, ["ao.num", "ao.num"], "↑-spin AO-basis Fock matrix (unrestricted)", violator=true),
    TrexioField("ao_1e_int", "fock_dn", Float64, ["ao.num", "ao.num"], "↓-spin AO-basis Fock matrix (unrestricted)", violator=true),
]

## 4.2a Positron orbitals (po group) - non-standard, violator
const TREXIO_PO_FIELDS = [
    TrexioField("po", "type", String, SCALAR, "free text to identify the set of POs (HF, Natural, Local, CASSCF, etc)", violator=true),
    TrexioField("po", "num", Int, SCALAR, "number of POs", violator=true),
    TrexioField("po", "coefficient", Float64, ["ao.num", "po.num"], "PO coefficients", violator=true),
    TrexioField("po", "coefficient_im", Float64, ["ao.num", "po.num"], "PO coefficients (imaginary part)", violator=true),
    TrexioField("po", "class", String, ["po.num"], "choose among: Core, Inactive, Active, Virtual, Deleted", violator=true),
    TrexioField("po", "symmetry", String, ["po.num"], "symmetry in the point group", violator=true),
    TrexioField("po", "occupation", Float64, ["po.num"], "occupation number", violator=true),
    TrexioField("po", "energy", Float64, ["po.num"], "for canonical POs, corresponding eigenvalue", violator=true),
    TrexioField("po", "spin", Int, ["po.num"], "for UHF wave functions, 0 is ↑ and 1 is ↓", violator=true),
    TrexioField("po", "k_point", Int, ["po.num"], "for periodic calculations, the k point to which each PO belongs", violator=true),
]

## 5.1a Extended determinant fields - non-standard
const TREXIO_DETERMINANT_EXT_FIELDS = [
    TrexioField("determinant", "n_int", Int, SCALAR, "number of 64-bit integers per spin pattern (ceil(mo.num/64))", violator=true),
    TrexioField("determinant", "alpha", Int, ["determinant.n_int", "determinant.num"], "alpha spin orbital patterns as 64-bit integer bit fields", violator=true),
    TrexioField("determinant", "beta", Int, ["determinant.n_int", "determinant.num"], "beta spin orbital patterns as 64-bit integer bit fields", violator=true),
    TrexioField("determinant", "coefficient_im", Float64, ["determinant.num"], "CI coefficients (imaginary part)", violator=true),
]

## 5.3a Dense amplitude fields - non-standard
const TREXIO_AMPLITUDE_EXT_FIELDS = [
    TrexioField("amplitude", "single_dense", Float64, ["v", "o"], "single excitation amplitudes (dense)", violator=true),
    TrexioField("amplitude", "single_dense_im", Float64, ["v", "o"], "single excitation amplitudes, imaginary part (dense)", violator=true),
    TrexioField("amplitude", "double_dense", Float64, ["v", "v", "o(o+1)/2"], "double excitation amplitudes (dense)", violator=true),
    TrexioField("amplitude", "double_dense_im", Float64, ["v", "v", "o(o+1)/2"], "double excitation amplitudes, imaginary part (dense)", violator=true),
    TrexioField("amplitude", "single_up_dense", Float64, ["v", "o"], "↑-spin component of the single excitation amplitudes (dense)", violator=true),
    TrexioField("amplitude", "single_up_dense_im", Float64, ["v", "o"], "↑-spin component of the single excitation amplitudes, imaginary part (dense)", violator=true),
    TrexioField("amplitude", "single_dn_dense", Float64, ["V", "O"], "↓-spin component of the single excitation amplitudes (dense)", violator=true),
    TrexioField("amplitude", "single_dn_dense_im", Float64, ["V", "O"], "↓-spin component of the single excitation amplitudes, imaginary part (dense)", violator=true),
    TrexioField("amplitude", "double_upup_dense", Float64, ["v(v-1)/2", "o(o-1)/2"], "↑↑-spin component of the double excitation amplitudes (dense)", violator=true),
    TrexioField("amplitude", "double_upup_dense_im", Float64, ["v(v-1)/2", "o(o-1)/2"], "↑↑-spin component of the double excitation amplitudes, imaginary part (dense)", violator=true),
    TrexioField("amplitude", "double_dndn_dense", Float64, ["V(V-1)/2", "O(O-1)/2"], "↓↓-spin component of the double excitation amplitudes (dense)", violator=true),
    TrexioField("amplitude", "double_dndn_dense_im", Float64, ["V(V-1)/2", "O(O-1)/2"], "↓↓-spin component of the double excitation amplitudes, imaginary part (dense)", violator=true),
    TrexioField("amplitude", "double_updn_dense", Float64, ["v", "V", "o", "O"], "↑↓-spin component of the double excitation amplitudes (dense)", violator=true),
    TrexioField("amplitude", "double_updn_dense_im", Float64, ["v", "V", "o", "O"], "↑↓-spin component of the double excitation amplitudes, imaginary part (dense)", violator=true),
]

# Combine all non-standard field definitions
const NONSTANDARD_TREXIO_FIELDS = vcat(
    TREXIO_BASIS_EXT_FIELDS,
    TREXIO_AO_1E_INT_EXT_FIELDS,
    TREXIO_PO_FIELDS,
    TREXIO_DETERMINANT_EXT_FIELDS,
    TREXIO_AMPLITUDE_EXT_FIELDS,
)