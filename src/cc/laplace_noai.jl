"""
    LaplaceQuadrature_NOAI module

This module provides functions to compute Laplace quadrature points and weights using the minimax (Remez) algorithm, 
as described by Takatsuka et al. in J. Chem. Phys. 129, 044112 (2008), but without determining nodal points to construct intervals.

"""
module LaplaceQuadrature_NOAI

using LinearAlgebra
export get_laplace_quadrature_noai

const LAPLACE_LIB = joinpath(@__DIR__, "..", "..", "lib", "laplace_points")




function get_laplace_quadrature_noai(εo::Vector{Float64}, εv::Vector{Float64}, npoints::Int; use_distribution::Bool=false)
    emin = 2*(minimum(εv) - maximum(εo))
    emax = 2*(maximum(εv) - minimum(εo))
        if emin <= 0
            error("Non-positive energy gap detected.")
        end 
    return calc_laplace_points(emin, emax, npoints)
end





"""
    calc_laplace_points(emin::Float64, emax::Float64, npoints::Int)

Compute Laplace quadrature points t and weights w.
"""
function calc_laplace_points(emin::Float64, emax::Float64, npoints::Int)

    # Initial guess for t: geometric progression or from table
  
    R = emax/emin
    #ACHTUNG! HIER DANIELS FUNKTION EINFUEGEN UND PRUEFEN, OB ZUERST W ODER T!!!!  
    w,t = get_initial_laplace_points(npoints, R; filename=LAPLACE_LIB)
    
    x = zeros(2*npoints+1)
    x[1] = 1
    x[2*npoints+1] = R
    maxiter = 100
    tol = 1e-12 
    
         
    Eta(x) = sum(w[k]*exp(-t[k]*x) for k=1:npoints) - 1/x 

    # Find nodal points of Eta(x) in (emin,emax) to set intervals
    roots = Float64[]
    scan = range(emin, emax, length=2000)
    prev = Eta(scan[1])
    for i in 2:length(scan)
        curr = Eta(scan[i])
        if prev*curr < 0
            # Bisection
            a, b = scan[i-1], scan[i]
            for _ in 1:50
                m = (a+b)/2
                v = Eta(m)
                if abs(v) < tol; break; end
                if Eta(a)*v < 0; b = m; else; a = m; end
            end
            push!(roots, (a+b)/2)
        end
        prev = curr
    end
    # Always use endpoints as reference points
    nodal_points_of_Eta = [1; sort(roots)[1:2*npoints-1]; R]
    
    if length(nodal_points_of_Eta) < 2*npoints+1
        println("ATTENTION! Not enough nodal points to span intervals!!!")
        break
    end
   


     
    for iter = 1:maxiter    
       
        #determine extrema within respective intervals with grid search
        gridsearch_pointnumber = 1000
       
        #only until 2*npoints since +1 is within the definiton of the interval limit 
        for i in 1:2*npoints
        
            interval_limit_left = nodal_points_of_Eta[i]
            interval_limit_right = nodal_points_of_Eta[i+1]
            interval_steps = abs(interval_limit_right - interval_limit_left)/gridsearch_pointnumber
            
            gridsearch_points_within_interval =  [interval_limit_left:interval steps:interval_limit_right;]
                        
            
            for i in gridsearch_points_within_interval
               
                if Eta(i)
                end           

            end          

        end  

        #Hier noch ergaenzen!!!
        if < tol
            break
        end


    end #iter


    return w, t

end





""" 
    get_initial_laplace_points(npoints::Int, delta::Float64; filename::String="lib/laplace_points")
    
Obtain initial guess for Laplace quadrature points from a table file.
- npoints: number of quadrature points
- delta: emax/emin
- filename: path to the table file (default: "lib/laplace_points")
    
Returns: Vector of points (t)
""" 
function get_initial_laplace_points(npoints::Int, delta::Float64; filename::String="lib/laplace_points")
    open(filename, "r") do io
        found = false
        t = Float64[]
        while !eof(io)
            line = readline(io)
            if startswith(line, "t_q") 
                parts = split(line)
                if length(parts) >= 3 && parse(Int, parts[2]) == npoints && parse(Float64, parts[3]) >= delta
                    # Read npoints numbers in the next line
                    found = true
                    t_line = readline(io)
                    t_parts = split(t_line)
                    t = Float64[]
                    for i in 1:npoints
                        push!(t, parse(Float64, t_parts[i]))
                    end
                    break
                end
            end
        end
        if !found
            error("No entry found in $(filename) for npoints=$(npoints), delta>=$(delta)")
        end
        return t
    end
end



end #module
