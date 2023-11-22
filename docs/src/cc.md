# Coupled-cluster methods

```@meta
CurrentModule = ElemCo.CoupledCluster
```

```@docs
CoupledCluster
```

Lagrange multiplier equations for coupled cluster singles/doubles methods:

```math
\begin{aligned}
\frac{\partial\mathcal{L}}{\partial T^m_e}&=
\left(2 v_{qm}^{pe} - v_{qm}^{ep}\right) \hat D_p^q + 2f_m^e 
- 2 Λ_{ij}^{eb} \hat v_{mb}^{ij}
+ 2 K_{mj}^{rs} \delta_r^e \left(\delta_s^j + \delta_s^b T^j_b \right) \\
&+2 D_{mj}^{kl} \hat v_{kl}^{ej}  
- 2 Λ_{ij}^{eb} \left(\hat v_{mb}^{cd} T^{ij}_{cd}\right)
- D_d^e \hat f_m^d + D_m^k \hat f_k^e 
- 2 D_{id}^{el} \hat v_{ml}^{id} 
+ 2 D_{md}^{al} \hat v_{al}^{ed}\\
&+ 2\bar D_{ic}^{ek} \hat v_{km}^{ic} 
- 2\bar D_{mc}^{ak} \hat v_{ka}^{ec} 
- Λ_{i}^{e} \hat f_{m}^{i}
+ Λ_{m}^{a} \hat f_{a}^{e}
- Λ_i^e x_m^i - Λ_m^a x_a^e.
\end{aligned}
```

```math
\begin{aligned}
\frac{\partial\mathcal{L}}{\partial T^{mn}_{ef}}&=
\tilde v_{mn}^{ef} 
+ Λ_{ij}^{ef} \left(\hat v_{mn}^{ij} \red{+ v_{mn}^{cd} T^{ij}_{cd}}\right) 
\red{+ D_{mn}^{kl} v_{kl}^{ef} } + K_{mn}^{rs} \delta_r^e \delta_s^f\\
&+ \mathcal{P}(em;fn)\left\{ 
Λ_{mn}^{af} \left(\hat f_a^e - \red{2\times}\frac{1}{2} x_a^e\right)
- Λ_{in}^{ef} \left(\hat f_m^i + \red{2\times}\frac{1}{2} x_m^i\right)
\right. \\
&+ \mathcal{T}(mn) \left[\red{2\times}\frac{1}{4} v_{kn}^{ef} D_m^k 
- \red{2\times}\frac{1}{4} v_{mn}^{cf} D_c^e
+ Λ_{in}^{af}\left(\hat v_{am}^{ie} + v_{km}^{ce}\tilde T^{ik}_{ac}\right)\right.\\
&\left.+ \frac{1}{2} \left(
  Λ_m^e \hat f_n^f 
+ Λ_n^a \hat v_{am}^{fe} - Λ_i^f \hat v_{nm}^{ie} \right) \right] 
\\
&\left.- Λ_{in}^{af} \hat v_{ma}^{ie} - Λ_{in}^{eb} \hat v_{mb}^{if}
\red{-D_{nc}^{fl} v_{ml}^{ce} +\bar D_{nd}^{ek}v_{km}^{fd}} \right\},\\
\end{aligned}
```
with
```math
\begin{aligned}
&K_{mn}^{rs} = \hat \Lambda_{mn}^{pq} v_{pq}^{rs} \\
&\hat \Lambda_{mn}^{pq} = Λ_{mn}^{ab}\delta_a^p\delta_b^q 
- Λ_{mn}^{ab} T^i_a  \delta_i^p \delta_b^q
- Λ_{mn}^{ab} \delta_a^p T^j_b \delta_j^q
+ Λ_{mn}^{ab} T^i_a T^j_b \delta_i^p \delta_j^q\\
&x_m^i = \tilde T^{il}_{cd} v_{ml}^{cd} \qquad\qquad
x_a^e = \tilde T^{kl}_{ac} v_{kl}^{ec}\\
&\mathcal{T}(mn) X_{mn}^{ef} = 2X_{mn}^{ef} - X_{nm}^{ef}\\
&D_{ij}^{kl} = \Lambda_{ij}^{cd} T^{kl}_{cd} \\
&D_{ib}^{aj} = \Lambda_{ik}^{ac} \tilde T^{kj}_{cb} \\
&\bar D_{ib}^{aj} = \Lambda_{ik}^{ac} T^{kj}_{cb} + \Lambda_{ik}^{ca} T^{kj}_{bc} \\
\end{aligned}
```

## Exported functions
 
```@autodocs
Modules = [CoupledCluster]
Private = false
Order = [:function]
``` 

## Internal functions
```@autodocs
Modules = [CoupledCluster]
Public = false
Order = [:function]
``` 
