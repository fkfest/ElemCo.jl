using Plots
using Colors
"""
  This is a logo generator.
  A coil in form of a letter m
"""
function generate_logo1()
  println("Generating logo...")
  t = 0.3:0.25:6π-0.5
  d=20
  fx(x) = sin(x)*d/(x+d)
  fy(x) = cos(x)*d/(x+d)
  fz(x) = x/(6π)*30/(x+30) - 0.5
  fms(x) = (3.0 +3*x/(6π))*(3+sin(x+π))/3
  x = reverse(fx.(t))
  y = reverse(fy.(t))
  z = reverse(fz.(t))
  ms = fms.(t)
  # plot(x, z, y, aspect_ratio=:equal,camera=(65,20), xlims=(-1, 1), zlims=(-1, 1), ylims=(-0.5, 0.5))
  scatter(x, z, y, aspect_ratio=:equal,camera=(70,20), xlim=(-1, 1), zlim=(-1, 1), ylim=(-0.5, 0.5), showaxis=false, ticks=false, linewidth=1, legend=false, markerstrokecolor=:steelblue, markersize=ms)
  png("coil.png")
end

function generate_logo2()
  println("Generating logo...")
  t = 0.3:0.27:6π-0.0
  d=20
  fx(x) = sin(x)*d/(x+d)
  fy(x) = cos(x)*d/(x+d)
  fz(x) = x/(6π)*30/(x+30) - 0.5
  fms(x) = (3.0 +3*x/(6π))*(3+sin(x+π))/3
  x = reverse(fx.(t))
  y = reverse(fy.(t))
  z = reverse(fz.(t))
  ms = fms.(t)
  # plot(x, z, y, aspect_ratio=:equal,camera=(65,20), xlims=(-1, 1), zlims=(-1, 1), ylims=(-0.5, 0.5))
  scatter(x, z, y, aspect_ratio=:equal,camera=(73,0), xlim=(-1, 1), zlim=(-1, 1), ylim=(-0.5, 0.5), showaxis=false, ticks=false, linewidth=1, legend=false, markerstrokecolor=:steelblue, markersize=ms)
  png("coil.png")
end

function generate_logo3()
  println("Generating logo...")
  t = 0.3:0.27:6π-0.0
  d=20
  fx(x) = sin(x)*d/(x+d)
  fy(x) = cos(x)*d/(x+d)
  fz(x) = x/(6π)*30/(x+30) - 0.5
  fms(x) = (3.0 +3*x/(6π))*(3+sin(x+π))/3
  x = reverse(fx.(t))
  y = reverse(fy.(t))
  z = reverse(fz.(t))
  ms = fms.(t)
  # plot(x, z, y, aspect_ratio=:equal,camera=(65,20), xlims=(-1, 1), zlims=(-1, 1), ylims=(-0.5, 0.5))
  scatter(x, z, y, aspect_ratio=:equal,camera=(180,0), xlim=(-1, 1), zlim=(-1, 1), ylim=(-0.5, 0.5), showaxis=false, ticks=false, linewidth=1, legend=false, markerstrokecolor=:steelblue, markersize=ms)
  png("coil.png")
end

function generate_logo4()
  println("Generating logo...")
  sc = 3.0
  last = 6π + 1.0
  d=9
  camera = (112,0)
  perspective(x) = (1.5 -x/last)*(3+sin(x+π/2))/8
  pers(x) = perspective(x)
  tt = -0.5
  α = 0.343
  t = Float64[]
  while tt < last
    push!(t, tt)
    tt += pers(tt)*α*(tt+20)/20
  end
  fms(x) = sc*15.0*perspective(x)
  fx(x) = sin(x-π/2)*d/(x+d)
  fy(x) = cos(x-π/2)*d/(x+d)
  fz(x) = x/(last)*35/(x+30) - 0.4
  x = reverse(fx.(t))
  y = reverse(fy.(t))
  z = reverse(fz.(t))
  ms = reverse(fms.(t))
  logocolors = Colors.JULIA_LOGO_COLORS
  println(length(t))
  mc = [logocolors.blue for i in 1:length(t)]
  #triangle!(mc, ms, logocolors)
  #line_above!(mc, ms, logocolors)
  #line_below!(mc, ms, logocolors)
  #random_colors!(mc, ms, logocolors)
  # periods!(mc, ms, logocolors)
  periods!(mc, ms, logocolors, [1, 2, 3, 1, 3, 2, 1, 2, 3])
  #shells!(mc, ms, logocolors)
  #periodic_table!(mc, ms, logocolors, 1)
  #overdraw!(mc,ms,x,y,z,[37, 38])
  # overdraw!(mc,ms,x,y,z,[55, 56, 87, 88])
  scatter(x, z, y, aspect_ratio=:equal,camera=camera,size=(sc*1200,sc*800), xlim=(-1.1, 1.1), zlim=(-1.0, 1), ylim=(-0.6, 0.5), showaxis=false, ticks=false, linewidth=1, legend=false, markercolor=mc, markerstrokecolor=mc, markersize=ms, background_color=:transparent)
  # gui()
  # readline()
  png("coil.png")
  run(`magick coil.png -trim coil.png`)
end

"""
  triangle of julia colors
"""
function triangle!(mc, ms, logocolors)
  mc[end] = logocolors.red
  mc[end-30] = logocolors.green
  mc[end-50] = logocolors.purple
  ms[end] *= 1.5
  ms[end-30] *= 1.5
  ms[end-50] *= 1.5
end

"""
  line of julia colors above
"""
function line_above!(mc, ms, logocolors)
  mc[end-6] = logocolors.red
  mc[end-31] = logocolors.green
  mc[end-56] = logocolors.purple
end

"""
  line of julia colors below
"""
function line_below!(mc, ms, logocolors)
  mc[end] = logocolors.red
  mc[end-25] = logocolors.green
  mc[end-50] = logocolors.purple
  ms[end] *= 1.5
  ms[end-25] *= 1.5
  ms[end-50] *= 1.5
end

"""
  random order of julia colors
"""
function random_colors!(mc, ms, logocolors)
  n = length(t) 
  ms_save = copy(ms)
  for i in 1:5
    for col in [logocolors.red, logocolors.green, logocolors.purple]
      ii = rand(1:n)
      mc[ii] = col
      ms[ii] = 1.5 * ms_save[ii]
    end
  end
end

"""
  julia colors at starts of periods in periodic table
"""
function periods!(mc, ms, logocolors, colororder=[])
  cols = [logocolors.red, logocolors.green, logocolors.purple]
  icol = 1
  for (ic, i) in enumerate([1,3,11,19,37,55,87,119])
    if i > length(mc)
      break
    end
    if length(colororder) > 0
      icol = colororder[ic]
    end
    mc[end-i+1] = cols[icol]
    if icol == 3
      icol = 1
    else
      icol += 1
    end
    ms[end-i+1] *= 1.7
  end
end

"""
  julia colors at starts of shells in periodic table
"""
function shells!(mc, ms, logocolors)
  cols = [logocolors.red, logocolors.green, logocolors.purple]
  icol = 1
  for i in [1,3,5,11,13,19,21,31,37,39,49,55,57,72,81,87,89,104,113]
    if i > length(mc)
      break
    end
    mc[end-i+1] = cols[icol]
    if icol == 3
      icol = 1
    else
      icol += 1
    end
    ms[end-i+1] *= 1.5
  end
end

function periodic_table!(mc, ms, logocolors, version=1)
  if version == 1
    nonmetals = [1:2; 6:10; 15:18; 34:36; 53:54; 85:86; 117:118]
    alkali = [3:4; 11:12; 19:20; 37:38; 55:56; 87:88]
    metals = [5:5; 13:14; 31:33; 49:52; 81:84; 113:116]
  else
    nonmetals = [1:2; 5:10; 14:18; 33:36; 52:54; 85:86; 117:118]
    alkali = [3:4; 11:12; 19:20; 37:38; 55:56; 87:88]
    metals = [13:13; 31:32; 49:51; 81:84; 113:116]
  end
  for i in nonmetals
    if i > length(mc)
      break
    end
    mc[end-i+1] = logocolors.green
  end
  for i in alkali
    if i > length(mc)
      break
    end
    mc[end-i+1] = logocolors.red
  end
  for i in metals
    if i > length(mc)
      break
    end
    mc[end-i+1] = logocolors.purple
  end
end

"""
  overdraw some points
"""
function overdraw!(mc, ms, x, y, z, points)
  n = length(mc)
  for p in points
    push!(mc, mc[n-p+1])
    push!(ms, ms[n-p+1])
    push!(x, x[n-p+1])
    push!(y, y[n-p+1])
    push!(z, z[n-p+1])
  end 
end

# generate_logo1()
# generate_logo2()
# generate_logo3()
generate_logo4()
