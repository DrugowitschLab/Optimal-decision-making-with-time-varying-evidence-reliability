## Plots the CIR process examples.
#
# All data is loaded from cir_example.jld

using Winston, Color

include("utils.jl")

# settings
datafile = "cir_examples.jld"
figpath = "figs"
# n colors as columns of returned matrix
pcols(n) = [RGB(xi, xi, xi) for xi in linspace(0.0, 0.7, n)]
pwidth = 2
yrange = [0, 1.6]


## Simulations for different mu_tau
@readdata(datafile, "var_mu_tau", mu_taus, sig_tau, theta_tau, tau_traj, ts)
p = FramedPlot(title = "CIR trajectories for different \\tau  means",
    xlabel = "t", ylabel = "\\tau",
    xrange = [ts[1], ts[end]], yrange = yrange,
    aspect_ratio = 3/4)
nmu_tau = length(mu_taus)
ccols = pcols(nmu_tau)
pcurve = cell(nmu_tau);
for i = 1:nmu_tau
    pcurve[i] = Curve(ts, tau_traj[i,:], color=ccols[i], width=pwidth)
    setattr(pcurve[i], label=@sprintf("%4.2f", mu_taus[i]))
    add(p, pcurve[i])
end
add(p, Legend(0.1, 0.9, pcurve))
display(p)
println("-- Figure saved as $figpath/cir_examples_mu_tau.eps")
savefig(p, joinpath(figpath, "cir_examples_mu_tau.eps"))


## Simulations for different sig_tau
@readdata(datafile, "var_sig_tau", mu_tau, sig_taus, theta_tau, tau_traj, ts)
p = FramedPlot(title = "CIR trajectories for different \\tau  SDs",
    xlabel = "t", ylabel = "\\tau",
    xrange = [ts[1], ts[end]], yrange = yrange,
    aspect_ratio = 3/4)
nsig_tau = length(sig_taus)
ccols = pcols(nsig_tau)
pcurve = cell(nsig_tau);
for i = 1:nsig_tau
    pcurve[i] = Curve(ts, tau_traj[i,:], color=ccols[i], width=pwidth)
    setattr(pcurve[i], label=@sprintf("%4.2f", sig_taus[i]))
    add(p, pcurve[i])
end
add(p, Legend(0.1, 0.9, pcurve))
display(p)
println("-- Figure saved as $figpath/cir_examples_sig_tau.eps")
savefig(p, joinpath(figpath, "cir_examples_sig_tau.eps"))


## Simulations for different theta_tau
@readdata(datafile, "var_theta_tau", mu_tau, sig_tau, theta_taus, tau_traj, ts)
p = FramedPlot(title = "CIR trajectories for different \\tau  speeds",
    xlabel = "t", ylabel = "\\tau",
    xrange = [ts[1], ts[end]], yrange = yrange,
    aspect_ratio = 3/4)
ntheta_tau = length(theta_taus)
ccols = pcols(ntheta_tau)
pcurve = cell(ntheta_tau);
for i = 1:ntheta_tau
    pcurve[i] = Curve(ts, tau_traj[i,:], color=ccols[i], width=pwidth)
    setattr(pcurve[i], label=@sprintf("%4.2f", theta_taus[i]))
    add(p, pcurve[i])
end
add(p, Legend(0.1, 0.9, pcurve))
display(p)
println("-- Figure saved as $figpath/cir_examples_theta_tau.eps")
savefig(p, joinpath(figpath, "cir_examples_theta_tau.eps"))
