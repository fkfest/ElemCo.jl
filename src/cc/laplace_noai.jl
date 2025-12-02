"""
    LaplaceQuadrature_NOAI module

This module provides functions to compute Laplace quadrature points and weights using the minimax (Remez) algorithm, 
as described by Takatsuka et al. in J. Chem. Phys. 129, 044112 (2008), but without determining nodal points to construct intervals.

"""
module LaplaceQuadrature_NOAI

using LinearAlgebra
export get_laplace_quadrature_noai

const LAPLACE_LIB_POINTS = joinpath(@__DIR__, "..", "..", "lib", "laplace_points")
const LAPLACE_LIB_WEIGHTS = joinpath(@__DIR__, "..", "..", "lib", "laplace_weights")



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
    t = get_initial_laplace_points_or_weights(npoints, R; filename=LAPLACE_LIB_POINTS)
    w = get_initial_laplace_points_or_weights(npoints, R; filename=LAPLACE_LIB_WEIGHTS)    

    tol = 1e-12 
    
         
    Eta(x) = sum(w[k]*exp(-t[k]*x) for k=1:npoints) - 1/x 
    d_Eta_dpoints(x,t_q,w_q) = -x * w_q * exp(-t_q * x) 
    d_Eta_dweights(x,t_q) = exp(-t_q*x)   

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
    nodal_points_of_Eta = [sort(roots)[1:2*npoints]]
    
    if length(nodal_points_of_Eta) < 2*npoints
        println("ATTENTION! Not enough nodal points to span intervals!!!")
        #break
    end
   

    maxiter = 100
    
    x_extrema_list = zeros(2*npoints-1)
    
    #das Nachfolgende ist falsch, da nur 2*npoints-1, oder??
    #x_extrema_list = zeros(2*npoints+1)
    #x_extrema_list[1] = 1
    #x_extrema_list[2*npoints+1] = R

    for iter = 1:maxiter    
         
        #determine extrema within respective intervals with grid search
        gridsearch_pointnumber = 1000
       
        #only until 2*npoints-1 since +1 is within the definiton of the interval limit 
        for i in 1:2*npoints-1
        
            interval_limit_left = nodal_points_of_Eta[i]
            interval_limit_right = nodal_points_of_Eta[i+1]
            interval_steps = abs(interval_limit_right - interval_limit_left)/gridsearch_pointnumber
            
            gridsearch_points_within_interval =  [interval_limit_left:interval_steps:interval_limit_right;]
            
            x_max_for_interval = interval_limit_left           
            Eta_old = 0
            for i in gridsearch_points_within_interval
                 
                if Eta(i) > Eta_old
                    Eta_old = Eta(i)
                    x_max_for_interval = i
                end           
                    
            end         
            push!(x_extrema_list,i)  
        end
       
        if length(x_extrema_list) < 2*npoints-1
            println("ATTENTION! Not enough extrema points!!!")
        end 
       
        #to get to 2k+1 x values, before only 2k-1 extrema were determined
        x_extrema_list_with_1_and_R = [1,x_extrema_list,R]         
  

       
        vector_weights_and_points_update = zeros(2*npoints) 
        #the gradient only has 2*npoints since it depends on x_i and x_(i+1) 
        #-> it matches the dimension of the number of weigths + points which is each npoints big
        #so the Hessian can be inverted
        gradient = zeros(2*npoints)
        Hessian = zeros(2*npoints,2*npoints) 

        for i in 1:2*npoints
            #evaluate Eta(x_i) + Eta(x_(i+1))
            gradient[i] = Eta(x_extrema_list_with_1_and_R[i]) + Eta(x_extrema_list_with_1_and_R[i+1])
        end 

        
        for i in 1:2*npoints
            for j in 1:npoints
                Hessian[i,j] = d_Eta_dpoints(x_extrema_list_with_1_and_R[i],t[j],w[j]) + d_Eta_dpoints(x_extrema_list_with_1_and_R[i+1],t[j],w[j])
                Hessian[i,j+npoints] = d_Eta_dweights(x_extrema_list_with_1_and_R[i],t[j]) + d_Eta_dweights(x_extrema_list_with_1_and_R[i+1],t[j])
            end
        end
        
        vector_weights_and_points_update = Hessian \ (-gradient)        

        #update t's and w's
        for i in 1:npoints
            t[i] += vector_weights_and_points_update[i] 
            w[i] += vector_weights_and_points_update[i+npoints]  
        end

        #define new eta
        Eta(x) = sum(w[k]*exp(-t[k]*x) for k=1:npoints) - 1/x 
                
        for i in 1:2*npoints
            #evaluate Eta(x_i) + Eta(x_(i+1)) for new set of t's and w's
            gradient[i] = Eta(x_extrema_list_with_1_and_R[i]) + Eta(x_extrema_list_with_1_and_R[i+1])
        end 

        #Hier noch ergaenzen!!!
        if sum(gradient,dims=1) < tol
            break
        end


    end #iter


    #rescale w's and t's

    for i in 1:2*npoints
        t[i] = t[i]/emin
        w[i] = w[i]/emin
    end


     
    return w, t

end





""" 
    get_initial_laplace_points_or_weights(npoints::Int, delta::Float64, filename::String)
    
Obtain initial guess for Laplace quadrature points from a table file.
- npoints: number of quadrature points
- delta: emax/emin
- filename: path to the table file
    
Returns: Vector of points (t)
""" 
function get_initial_laplace_points_or_weights(npoints::Int, delta::Float64, filename::String)
    open(filename, "r") do io
        found = false
        t = Float64[]
        while !eof(io)
            line = readline(io)
            if startswith(line, "n_q") 
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
