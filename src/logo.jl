using Plots
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
  last = 6π
  d=9
  t = -0.5:0.25:last+1.0
  fx(x) = sin(x-π/2)*d/(x+d)
  fy(x) = cos(x-π/2)*d/(x+d)
  fz(x) = x/(6π)*30/(x+30) - 0.4
  fms(x) = 1.6(3.0 +3*x/(last))*(3+sin(x+π/2))/3
  # x = (fx.(t))
  # y = (fy.(t))
  # z = (fz.(t))
  # ms = reverse(fms.(t))
  x = reverse(fx.(t))
  y = reverse(fy.(t))
  z = reverse(fz.(t))
  ms = (fms.(t))
  # plot(x, z, y, aspect_ratio=:equal,camera=(65,20), xlims=(-1, 1), zlims=(-1, 1), ylims=(-0.5, 0.5))
  scatter(x, z, y, aspect_ratio=:equal,camera=(110,0),size=(1200,800), xlim=(-1.0, 1), zlim=(-1.0, 1), ylim=(-0.6, 0.5), showaxis=false, ticks=false, linewidth=1, legend=false, markerstrokecolor=:steelblue, markersize=ms)
  # gui()
  # readline()
  png("coil.png")
end

# generate_logo1()
# generate_logo2()
# generate_logo3()
generate_logo4()