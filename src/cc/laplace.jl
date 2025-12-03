
"""
    LaplaceQuadrature module

This module provides functions to compute Laplace quadrature points and weights
for MP2 calculations using the minimax (Remez) algorithm, as described by Takatsuka et al. in J. Chem. Phys. 129, 044112 (2008)
or using the Häser-Almlöf simplex method.

It includes functions for both unweighted and weighted Laplace quadrature, where the latter uses a distribution function to improve accuracy.
"""
module LaplaceQuadrature

using LinearAlgebra
export get_laplace_quadrature, get_laplace_quadrature_simplex

const LAPLACE_LIB_POINTS = joinpath(@__DIR__, "..", "..", "lib", "laplace_points")
const LAPLACE_LIB_WEIGHTS = joinpath(@__DIR__, "..", "..", "lib", "laplace_weights")

"""
    calc_laplace_points(emin::Float64, emax::Float64, npoints::Int)

Compute Laplace quadrature points and weights for Laplace MP2 using the minimax (Remez) algorithm
from Takatsuka et al., J. Chem. Phys. 129, 044112 (2008).
Returns (weights, points).
"""
function calc_laplace_points(emin::Float64, emax::Float64, npoints::Int)

  # Initial guess for t: geometric progression or from table

  R = emax/emin
  #ACHTUNG! HIER DANIELS FUNKTION EINFUEGEN UND PRUEFEN, OB ZUERST W ODER T!!!!  
  t = get_initial_laplace(npoints, R, LAPLACE_LIB_POINTS)
  w = get_initial_laplace(npoints, R, LAPLACE_LIB_WEIGHTS)

  tol = 1e-6
  tol_newton = 1e-10

  Eta(x) = sum(w[k]*exp(-t[k]*x) for k=1:npoints) - 1/x
  d_Eta_dpoints(x,t_q,w_q) = -x * w_q * exp(-t_q * x)
  d_Eta_dweights(x,t_q) = exp(-t_q*x)
  d_Eta_dx(x) = - sum(t[k]*w[k]*exp(-t[k]*x) for k=1:npoints) + 1/(x^2) 
 
  #evaluate Eta(x_i) + Eta(x_(i+1))
  gradient(xi1,xi2) = Eta(xi1) + Eta(xi2)

  #starting guess for extrema in interval 1,R 
  #x_extrema_list = [cos((pi*(j-1))/(2*npoints)) *(R-1)/2 + (R+1)/2 for j in 1:(2*npoints+1)]
  
  x_extrema_list =  determine_nodal_points(d_Eta_dx,R,tol)
  x_extrema_list = sort(x_extrema_list)
  x_extrema_list = cat(1,x_extrema_list,R,dims=1)
  println(x_extrema_list)

  println(size(x_extrema_list))
  println(2*npoints+1)

  if length(x_extrema_list) < 2*npoints+1
    println("ATTENTION! Not enough extrema!!!")
    #break
  end    

  maxiter = 100
  for iter = 1:maxiter
    print("iter:")
    println(iter)
 
    vector_weights_and_points_update = zeros(2*npoints) 
    #the gradient only has 2*npoints since it depends on x_i and x_(i+1) 
    #-> it matches the dimension of the number of weigths + points which is each npoints big
    #so the Hessian can be inverted
    gradient_vector = zeros(2*npoints)
    Hessian = zeros(2*npoints,2*npoints) 
    for it_newton in 1:maxiter
      for i in 1:2*npoints
        #evaluate Eta(x_i) + Eta(x_(i+1))
        gradient_vector[i] = gradient(x_extrema_list[i],x_extrema_list[i+1])
      end

      for i in 1:2*npoints
        for j in 1:npoints
          Hessian[i,j] = d_Eta_dpoints(x_extrema_list[i],t[j],w[j]) + d_Eta_dpoints(x_extrema_list[i+1],t[j],w[j])
          Hessian[i,j+npoints] = d_Eta_dweights(x_extrema_list[i],t[j]) + d_Eta_dweights(x_extrema_list[i+1],t[j])
        end
      end

      println(det(Hessian))

      vector_weights_and_points_update = Hessian \ gradient_vector

      println("Test") 

      #update t's and w's
      for i in 1:npoints
        t[i] -= vector_weights_and_points_update[i] 
        w[i] -= vector_weights_and_points_update[i+npoints]  
      end

      for i in 1:2*npoints
        #evaluate Eta(x_i) + Eta(x_(i+1))
        gradient_vector[i] = gradient(x_extrema_list[i],x_extrema_list[i+1])
      end

      #Hier noch ergaenzen!!!
      if sum(abs, gradient_vector) < tol_newton
        break
      end

    end #iters of Newton-Raphson
    
    # Find nodal points of Eta(x) in (emin,emax) to set intervals
    
   # roots = Float64[]
   # #scan = range(emin, emax, length=10000)
   # scan = range(1, R, length=10000)
   # prev = Eta(scan[1])
   # for i in 2:length(scan)
   #   curr = Eta(scan[i])
   #   if prev*curr < 0
   #     # Bisection
   #     a, b = scan[i-1], scan[i]
   #     for _ in 1:200
   #       m = (a+b)/2
   #       v = Eta(m)
   #       if abs(v) < tol; break; end
   #       if Eta(a)*v < 0; b = m; else; a = m; end
   #     end
   #     push!(roots, (a+b)/2)
   #   end
   #   prev = curr
   # end
    
    roots = determine_nodal_points(Eta,R,tol)

    if length(roots) < 2*npoints
      println("ATTENTION! Not enough nodal points to span intervals!!!")
      #break
    end
    
    # Always use endpoints as reference points
    nodal_points_of_Eta = sort(roots)[1:2*npoints]

    #determine extrema within respective intervals with grid search
    gridsearch_pointnumber = 1000

    x_extrema_list = [1.0]  #start with 1.0 as first point
    #only until 2*npoints-1 since +1 is within the definiton of the interval limit
    for i in 1:2*npoints-1

      interval_limit_left = nodal_points_of_Eta[i]
      interval_limit_right = nodal_points_of_Eta[i+1]
      interval_steps = abs(interval_limit_right - interval_limit_left)/gridsearch_pointnumber

      x_max_for_interval = interval_limit_left
      Eta_old = 0
      for xi in interval_limit_left:interval_steps:interval_limit_right

        if abs(Eta(xi)) > Eta_old
          Eta_old = abs(Eta(xi))
          x_max_for_interval = xi
        end

      end
      push!(x_extrema_list, x_max_for_interval)
    end

    if length(x_extrema_list) < 2*npoints-1
      println("ATTENTION! Not enough extrema points!!!")
    end

    #to get to 2k+1 x values, before only 2k-1 extrema were determined
    push!(x_extrema_list, R)

    gradient_vector = zeros(2*npoints)
    for i in 1:2*npoints
      #evaluate Eta(x_i) + Eta(x_(i+1))
      gradient_vector[i] = gradient(x_extrema_list[i],x_extrema_list[i+1])
    end

    
    println("Test2")
    if norm(gradient_vector) < tol
      println("Converged!")
      break
    end
        
  end #iter


  #rescale w's and t's
  
  
  t_transformed = zeros(npoints)
  w_transformed = zeros(npoints)
  for i in 1:npoints
    t_transformed[i] = t[i]/emin
    w_transformed[i] = w[i]/emin
  end
  println(t_transformed)
  println(w_transformed)
  return w_transformed, t_transformed
end


function determine_nodal_points(f::Function,R::Float64,tol::Float64)
  roots = Float64[]
  #scan = range(emin, emax, length=10000)
  scan = range(1, R, length=10000)
  prev = f(scan[1])
  for i in 2:length(scan)
    curr = f(scan[i])
    if prev*curr < 0
      # Bisection
      a, b = scan[i-1], scan[i]
      for _ in 1:200
        m = (a+b)/2
        v = f(m)
        if abs(v) < tol; break; end
        if f(a)*v < 0; b = m; else; a = m; end
      end
      push!(roots, (a+b)/2)
    end
    prev = curr
  end
  return roots
end


"""
    build_distribution_function(εo::Vector{Float64}, εv::Vector{Float64})

Build the distribution function f(x) for energy denominators following Häser-Almlöf.
Creates histograms of Δᵢᵃ = εₐ - εᵢ using 600 bins, then combines them into 1200 bins.

Returns:
- x_grid: Grid points for the distribution (1200 points)
- f_x: Distribution function values at grid points
- xmin, xmax: Range of energy denominators
"""
function build_distribution_function(εo::Vector{Float64}, εv::Vector{Float64})
  # Calculate all energy denominators Δᵢᵃ = εₐ - εᵢ
  Δ = [εv[a] - εo[i] for i in eachindex(εo), a in eachindex(εv)]
  Δ_vec = vec(Δ)
  
  xmin = minimum(Δ_vec)
  xmax = maximum(Δ_vec)
  
  # First histogram: 600 bins for individual Δᵢᵃ
  nbins_single = 600
  bin_edges = range(xmin, xmax, length=nbins_single+1)
  hist1 = zeros(nbins_single)
  
  # Fill histogram with Δᵢᵃ values
  for δ in Δ_vec
    bin_idx = min(nbins_single, max(1, Int(floor((δ - xmin) / (xmax - xmin) * nbins_single)) + 1))
    hist1[bin_idx] += 1.0
  end
  
  # Second histogram: same as first (representing Δⱼᵇ)
  hist2 = copy(hist1)
  
  # Combined histogram: 1200 bins
  nbins_total = 1200
  x_grid = zeros(nbins_total)
  f_x = zeros(nbins_total)
  
  # Create logarithmic grid for final distribution
  log_xmin = log(xmin)
  log_xmax = log(xmax)
  for i in 1:nbins_total
    log_x = log_xmin + (log_xmax - log_xmin) * (i - 0.5) / nbins_total
    x_grid[i] = exp(log_x)
  end
  
  # Interpolate and combine histograms
  bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in 1:nbins_single]
  
  for i in 1:nbins_total
    x = x_grid[i]
    # Linear interpolation of both histograms
    f1 = interpolate_linear(bin_centers, hist1, x)
    f2 = interpolate_linear(bin_centers, hist2, x)
    f_x[i] = f1 + f2
  end
  
  # Normalize distribution
  total = sum(f_x)
  if total > 0
    f_x ./= total
  else
    f_x .= 1.0 / nbins_total
  end
  
  return x_grid, f_x, xmin, xmax
end

"""
    interpolate_linear(x_data::Vector{Float64}, y_data::Vector{Float64}, x::Float64)

Linear interpolation helper function.
"""
function interpolate_linear(x_data::Vector{Float64}, y_data::Vector{Float64}, x::Float64)
  if x <= x_data[1]
    return y_data[1]
  elseif x >= x_data[end]
    return y_data[end]
  end
  
  # Find bracketing indices
  i = 1
  while i < length(x_data) && x_data[i+1] < x
    i += 1
  end
  
  if i == length(x_data)
    return y_data[end]
  end
  
  # Linear interpolation
  t = (x - x_data[i]) / (x_data[i+1] - x_data[i])
  return y_data[i] * (1 - t) + y_data[i+1] * t
end

"""
    select_weighted_reference_points(x_grid::Vector{Float64}, f_x::Vector{Float64}, nref::Int, emin::Float64, emax::Float64)

Select initial reference points using distribution function weighting.
"""
function select_weighted_reference_points(x_grid::Vector{Float64}, f_x::Vector{Float64}, nref::Int, emin::Float64, emax::Float64)
  # Start with Chebyshev nodes as base points
  x_cheb = [0.5*(emin+emax) + 0.5*(emax-emin)*cos(pi*(2*j-1)/(2*nref)) for j in 1:nref]
  sort!(x_cheb)
  
  # Ensure endpoints are included and points are well-separated
  x_ref = Float64[]
  push!(x_ref, emin)
  
  # Add interior points, ensuring minimum separation
  min_sep = (emax - emin) / (2 * nref)
  for i in 2:nref-1
    x_candidate = x_cheb[i]
    # Ensure minimum separation from existing points
    while any(abs(x_candidate - x) < min_sep for x in x_ref)
      x_candidate += min_sep * (0.5 - rand())
    end
    x_candidate = max(emin + min_sep, min(emax - min_sep, x_candidate))
    push!(x_ref, x_candidate)
  end
  
  push!(x_ref, emax)
  
  return sort!(x_ref)
end

"""
    interpolate_distribution(x::Float64, x_grid::Vector{Float64}, f_x::Vector{Float64})

Interpolate distribution function value at point x.
"""
function interpolate_distribution(x::Float64, x_grid::Vector{Float64}, f_x::Vector{Float64})
  if x <= x_grid[1]
    return f_x[1]
  elseif x >= x_grid[end]
    return f_x[end]
  end
  
  # Linear interpolation
  i = searchsortedfirst(x_grid, x) - 1
  i = max(1, min(length(x_grid)-1, i))
  
  t = (x - x_grid[i]) / (x_grid[i+1] - x_grid[i])
  return (1-t) * f_x[i] + t * f_x[i+1]
end

"""
    find_weighted_extrema(E::Function, dE::Function, x_grid::Vector{Float64}, f_x::Vector{Float64}, 
                         emin::Float64, emax::Float64, nroots::Int, tol::Float64)

Find extrema of the error function weighted by the distribution function.
"""
function find_weighted_extrema(E::Function, dE::Function, x_grid::Vector{Float64}, f_x::Vector{Float64}, 
                            emin::Float64, emax::Float64, nroots::Int, tol::Float64)
  roots = Float64[]
  scan = range(emin, emax, length=2000)
  
  # Weighted derivative: f(x) * dE(x)
  weighted_dE(x) = interpolate_distribution(x, x_grid, f_x) * dE(x)
  
  prev = weighted_dE(scan[1])
  for i in 2:length(scan)
    curr = weighted_dE(scan[i])
    if prev*curr < 0
      # Bisection to find root
      a, b = scan[i-1], scan[i]
      for _ in 1:50
        m = (a+b)/2
        v = weighted_dE(m)
        if abs(v) < tol; break; end
        if weighted_dE(a)*v < 0; b = m; else; a = m; end
      end
      push!(roots, (a+b)/2)
    end
    prev = curr
  end
  
  return sort(roots)[1:min(nroots, length(roots))]
end

"""
    sample_from_distribution(x_grid::Vector{Float64}, f_x::Vector{Float64}, emin::Float64, emax::Float64)

Sample a point from the distribution function (inverse transform sampling).
"""
function sample_from_distribution(x_grid::Vector{Float64}, f_x::Vector{Float64}, emin::Float64, emax::Float64)
  # Find grid points within [emin, emax]
  mask = (x_grid .>= emin) .& (x_grid .<= emax)
  x_sub = x_grid[mask]
  f_sub = f_x[mask]
  
  if isempty(x_sub)
    return emin + (emax - emin) * rand()
  end
  
  # Cumulative distribution
  cum_f = cumsum(f_sub)
  cum_f ./= cum_f[end]
  
  # Sample
  r = rand()
  idx = searchsortedfirst(cum_f, r)
  idx = max(1, min(length(x_sub), idx))
  
  return x_sub[idx]
end

"""
    compute_weighted_barycentric_weights(x_ref::Vector{Float64}, x_grid::Vector{Float64}, f_x::Vector{Float64})

Compute barycentric weights considering distribution function.
"""
function compute_weighted_barycentric_weights(x_ref::Vector{Float64}, x_grid::Vector{Float64}, f_x::Vector{Float64})
  n = length(x_ref)
  beta = zeros(n)
  
  for j in 1:n
    weight_j = interpolate_distribution(x_ref[j], x_grid, f_x)
    denom = prod(abs(x_ref[j] - x_ref[l]) for l in 1:n if l != j)
    beta[j] = weight_j * (-1)^(j-1) / denom
  end
  
  return beta
end

"""
    find_roots(H::Function, tmin::Float64, tmax::Float64, nroots::Int, tol::Float64)

Find roots of function H in interval [tmin, tmax].
"""
function find_roots(H::Function, tmin::Float64, tmax::Float64, nroots::Int, tol::Float64)
  roots = Float64[]
  scan = range(tmin, tmax, length=2000)
  
  prev = H(scan[1])
  for i in 2:length(scan)
    curr = H(scan[i])
    if prev*curr < 0
      # Bisection
      a, b = scan[i-1], scan[i]
      for _ in 1:50
        m = (a+b)/2
        v = H(m)
        if abs(v) < tol; break; end
        if H(a)*v < 0; b = m; else; a = m; end
      end
      push!(roots, (a+b)/2)
    end
    prev = curr
  end
  
  return sort(roots)[1:min(nroots, length(roots))]
end

"""
    get_initial_laplace(npoints::Int, delta::Float64, filename::String)

Obtain initial guess for Laplace quadrature points from a table file.
- npoints: number of quadrature points
- delta: emax/emin
- filename: path to the table file

Returns: Vector of points (t)
"""
function get_initial_laplace(npoints::Int, delta::Float64, filename::String)
  open(filename, "r") do io
    found = false
    t = Float64[]
    while !eof(io)
      line = readline(io)
      if startswith(line, "n_q")
        parts = split(line)
        if length(parts) >= 3 && parse(Int, parts[2]) == npoints && parse(Float64, parts[3]) <= delta
          # Read npoints numbers in the next line
          found = true
          t_line = readline(io)
          t_parts = split(t_line)
          t = Float64[] 
          for i in 1:npoints
            push!(t, parse(Float64, t_parts[i]))
          end
          if parse(Int, parts[2]) > npoints || (parse(Int, parts[2]) == npoints && parse(Float64, parts[3]) > delta)
            break
          end
        end
      end
    end
    if !found
      error("No entry found in $(filename) for npoints=$(npoints), delta>=$(delta)")
    end
    return t
  end
end

"""
    calculate_weights(t_points::Vector{Float64}, x_grid::Vector{Float64}, f_x::Vector{Float64})

Calculate optimal weights for given t_points by solving the linear least-squares system.
The weights minimize ∫ f(x) * (1/x - Σq wq * exp(-x*tq))² dx.

Returns:
- w: Optimal weights
"""
function calculate_weights(t_points::Vector{Float64}, x_grid::Vector{Float64}, f_x::Vector{Float64})
  npoints = length(t_points)
  ngrid = length(x_grid)
  
  # Build linear system: A * w = b
  # where A[i,j] = ∫ f(x) * exp(-x*tᵢ) * exp(-x*tⱼ) dx
  # and   b[i]   = ∫ f(x) * exp(-x*tᵢ) * (1/x) dx
  
  A = zeros(npoints, npoints)
  b = zeros(npoints)
  
  # Numerical integration using trapezoidal rule
  for k in 1:ngrid-1
    x1, x2 = x_grid[k], x_grid[k+1]
    f1, f2 = f_x[k], f_x[k+1]
    dx = x2 - x1
    
    # Trapezoidal rule weights
    w1, w2 = dx/2, dx/2
    
    for i in 1:npoints
      exp_i1 = exp(-x1 * t_points[i])
      exp_i2 = exp(-x2 * t_points[i])
      
      # b vector: ∫ f(x) * exp(-x*tᵢ) * (1/x) dx
      b[i] += w1 * f1 * exp_i1 / x1 + w2 * f2 * exp_i2 / x2
      
      for j in 1:npoints
        exp_j1 = exp(-x1 * t_points[j])
        exp_j2 = exp(-x2 * t_points[j])
        
        # A matrix: ∫ f(x) * exp(-x*tᵢ) * exp(-x*tⱼ) dx
        A[i,j] += w1 * f1 * exp_i1 * exp_j1 + w2 * f2 * exp_i2 * exp_j2
      end
    end
  end
  
  # Solve linear system
  try
    w = A \ b
    return w
  catch
    # Fallback to pseudo-inverse if singular
    return pinv(A) * b
  end
end

"""
    objective_function(t_points::Vector{Float64}, x_grid::Vector{Float64}, f_x::Vector{Float64})

Compute the least-squares objective function for given t_points.
Returns the value of ∫ f(x) * (1/x - Σq wq * exp(-x*tq))² dx.
"""
function objective_function(t_points::Vector{Float64}, x_grid::Vector{Float64}, f_x::Vector{Float64})
  w = calculate_weights(t_points, x_grid, f_x)
  
  objective = 0.0
  ngrid = length(x_grid)
  
  # Numerical integration using trapezoidal rule
  for k in 1:ngrid-1
    x1, x2 = x_grid[k], x_grid[k+1]
    f1, f2 = f_x[k], f_x[k+1]
    dx = x2 - x1
    
    # Evaluate error at grid points
    sum1 = sum(w[i] * exp(-x1 * t_points[i]) for i in eachindex(t_points))
    sum2 = sum(w[i] * exp(-x2 * t_points[i]) for i in eachindex(t_points))
    
    error1 = (1/x1 - sum1)^2
    error2 = (1/x2 - sum2)^2
    
    # Trapezoidal integration
    objective += (dx/2) * (f1 * error1 + f2 * error2)
  end
  
  return objective
end

"""
    nelder_mead_simplex(f, x0::Vector{Float64}; 
                        maxiter=1000, tol=1e-8, α=1.0, β=0.5, γ=2.0, δ=0.5)

Simple Nelder-Mead simplex optimization.
- f: objective function
- x0: initial point
- Returns: optimized point
"""
function nelder_mead_simplex(f, x0::Vector{Float64}; 
                           maxiter=1000, tol=1e-8, α=1.0, β=0.5, γ=2.0, δ=0.5)
  n = length(x0)
  
  # Initialize simplex
  simplex = Vector{Vector{Float64}}(undef, n+1)
  simplex[1] = copy(x0)
  
  for i in 2:n+1
    simplex[i] = copy(x0)
    simplex[i][i-1] += 0.1 * (abs(x0[i-1]) + 0.1)
  end
  
  # Evaluate function at simplex vertices
  f_vals = [f(x) for x in simplex]
  
  for iter in 1:maxiter
    # Sort vertices
    perm = sortperm(f_vals)
    simplex = simplex[perm]
    f_vals = f_vals[perm]
    
    # Check convergence
    if f_vals[end] - f_vals[1] < tol
      break
    end
    
    # Centroid of best n points
    centroid = sum(simplex[1:end-1]) / n
    
    # Reflection
    x_r = centroid + α * (centroid - simplex[end])
    f_r = f(x_r)
    
    if f_vals[1] <= f_r < f_vals[end-1]
      # Accept reflection
      simplex[end] = x_r
      f_vals[end] = f_r
    elseif f_r < f_vals[1]
      # Expansion
      x_e = centroid + γ * (x_r - centroid)
      f_e = f(x_e)
      if f_e < f_r
        simplex[end] = x_e
        f_vals[end] = f_e
      else
        simplex[end] = x_r
        f_vals[end] = f_r
      end
    else
      # Contraction
      if f_r < f_vals[end]
        # Outside contraction
        x_c = centroid + β * (x_r - centroid)
      else
        # Inside contraction
        x_c = centroid + β * (simplex[end] - centroid)
      end
      f_c = f(x_c)
      
      if f_c < min(f_r, f_vals[end])
        simplex[end] = x_c
        f_vals[end] = f_c
      else
        # Shrink
        for i in 2:n+1
          simplex[i] = simplex[1] + δ * (simplex[i] - simplex[1])
          f_vals[i] = f(simplex[i])
        end
      end
    end
  end
  
  # Return best point
  best_idx = argmin(f_vals)
  return simplex[best_idx]
end

"""
    calc_laplace_points_simplex(εo::Vector{Float64}, εv::Vector{Float64}, npoints::Int;
                               include_zero::Bool=true, maxiter::Int=1000, tol::Float64=1e-8)

Calculate Laplace quadrature points and weights using the Häser-Almlöf simplex method.

Parameters:
- εo: Occupied orbital energies
- εv: Virtual orbital energies  
- npoints: Total number of quadrature points
- include_zero: Whether to include and fix t=0 point
- maxiter: Maximum iterations for simplex optimization
- tol: Convergence tolerance

Returns:
- Tuple (weights, points): wq and tq for Laplace quadrature
"""
function calc_laplace_points_simplex(εo::Vector{Float64}, εv::Vector{Float64}, npoints::Int;
                                   include_zero::Bool=true, maxiter::Int=1000, tol::Float64=1e-8)
    
  # Build distribution function
  x_grid, f_x, xmin, xmax = build_distribution_function(εo, εv)
  
  if include_zero
    # Optimize npoints-1 non-zero points, keep t=0 fixed
    n_optimize = npoints - 1
    
    # Initial guess for non-zero points (geometric progression)
    if n_optimize > 0
      t_init = [xmin * (xmax/xmin)^((k-1)/(n_optimize-1)) for k in 1:n_optimize]
    else
      t_init = Float64[]
    end
    
    # Objective function that includes t=0
    function obj_with_zero(t_vars::Vector{Float64})
      t_full = [0.0; t_vars]
      return objective_function(t_full, x_grid, f_x)
    end
    
    if n_optimize > 0
      # Optimize using simplex
      t_opt = nelder_mead_simplex(obj_with_zero, t_init; maxiter=maxiter, tol=tol)
      t_points = [0.0; t_opt]
    else
      t_points = [0.0]
    end
  else
    # Optimize all points
    # Initial guess (geometric progression)
    t_init = get_initial_laplace(npoints, xmax-xmin, LAPLACE_LIB_POINTS)
    # t_init = [xmin * (xmax/xmin)^((k-1)/(npoints-1)) for k in 1:npoints]
    
    # Objective function
    obj(t_vars) = objective_function(t_vars, x_grid, f_x)
    
    # Optimize using simplex
    t_points = nelder_mead_simplex(obj, t_init; maxiter=maxiter, tol=tol)
  end
  
  # Calculate final weights
  w = calculate_weights(t_points, x_grid, f_x)
  
  return w, t_points
end

"""
    get_laplace_quadrature_simplex(εo::Vector{Float64}, εv::Vector{Float64}, npoints::Int)

Wrapper function for simplex-based Laplace quadrature calculation.
"""
function get_laplace_quadrature_simplex(εo::Vector{Float64}, εv::Vector{Float64}, npoints::Int)
  emin = minimum(εv) - maximum(εo)
  if emin <= 0
    error("Non-positive energy gap detected.")
  end
  
  return calc_laplace_points_simplex(εo, εv, npoints; include_zero=false)
end

"""
    get_laplace_quadrature(εo::Vector{Float64}, εv::Vector{Float64}, npoints::Int; use_distribution::Bool=false)

Given occupied and virtual orbital energies, compute Laplace quadrature points and weights.
When use_distribution=true, uses distribution function weighting for improved accuracy.

Returns:
- Tuple (weights, points): wq and tq for Laplace quadrature
"""
function get_laplace_quadrature(εo::Vector{Float64}, εv::Vector{Float64}, npoints::Int; use_distribution::Bool=false)
  if use_distribution
    error("Weighted Laplace quadrature not yet implemented in this function.")
    # return calc_laplace_points_weighted(εo, εv, npoints; use_distribution=true)
  else
    emin = 2 * (minimum(εv) - maximum(εo))
    emax = 2 * (maximum(εv) - minimum(εo))
    if emin <= 0
      error("Non-positive energy gap detected.")
    end
    return calc_laplace_points(emin, emax, npoints)
  end
end

end # module
