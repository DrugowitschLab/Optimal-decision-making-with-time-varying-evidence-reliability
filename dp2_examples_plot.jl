# plots data from dp2_examples.jl

using Winston, Color

using VI2, CIRProc

include("utils.jl")

# settings
datafile = "dp2_examples.jld"
figpath = "figs"
# n colors as columns of returned matrix
pcols(n) = [RGB(xi, xi, xi) for xi in linspace(0.0, 0.7, n)]
pwidth = 5


## Bounds for different costs
@readdata(datafile, "var_c", mu_tau, sig_tau, theta_tau, cs, bound, taus)
cirp = CIRProcess(mu_tau, sig_tau, theta_tau)
ptau = pdf(ssdist(cirp), taus)
p = FramedPlot(title="Optimal bound for different costs",
    xlabel = "\\tau", ylabel = "g_\\theta(\\tau)",
    xrange = [taus[1], taus[div(length(taus),2)]], yrange = [0.5, 1.0],
    aspect_ratio = 3/4)
nc = length(cs)
ccols = pcols(nc)
add(p,Curve(taus, 0.5.+0.2*ptau/maximum(ptau), kind="dashed"))
pcurve = cell(nc);
for i = 1:nc
    pcurve[i] = Curve(taus, bound[i,:], color=ccols[i], width=pwidth)
    setattr(pcurve[i], label=@sprintf("%4.2f", cs[i]))
    add(p, pcurve[i])
end
add(p, Legend(0.1, 0.9, pcurve))
display(p)
println("-- Figure saved as $figpath/dp2_examples_c.eps")
savefig(p, joinpath(figpath, "dp2_examples_c.eps"))


## Bound for different mu_tau
@readdata(datafile, "var_mu_tau", mu_taus, sig_tau, theta_tau, bound, taus)
p = FramedPlot(title = "Optimal bound for different \\tau  means",
    xlabel = "\\tau", ylabel = "g_\\theta(\\tau)",
    xrange = [taus[1], taus[div(length(taus),2)]], yrange = [0.5, 1.0],
    aspect_ratio = 3/4)
nmu_tau = length(mu_taus)
ccols = pcols(nmu_tau)
for i = 1:nmu_tau
    cirp = CIRProcess(mu_taus[i], sig_tau, theta_tau)
    ptau = pdf(ssdist(cirp), taus)
    add(p, Curve(taus, 0.5.+0.1*ptau, color=ccols[i], kind="dashed"))
end
pcurve = cell(nmu_tau);
for i = 1:nmu_tau
    pcurve[i] = Curve(taus, bound[i,:], color=ccols[i], width=pwidth)
    setattr(pcurve[i], label=@sprintf("%4.2f", mu_taus[i]))
    add(p, pcurve[i])
end
add(p, Legend(0.1, 0.9, pcurve))
display(p)
println("-- Figure saved as $figpath/dp2_examples_mu_tau.eps")
savefig(p, joinpath(figpath, "dp2_examples_mu_tau.eps"))


## Bound for different sig_tau
@readdata(datafile, "var_sig_tau", mu_tau, sig_taus, theta_tau, bound, taus)
p = FramedPlot(title = "Optimal bound for different \\tau  SDs",
    xlabel = "\\tau", ylabel = "g_\\theta(\\tau)",
    xrange = [taus[1], taus[div(length(taus),2)]], yrange = [0.5, 1.0],
    aspect_ratio = 3/4)
nsig_tau = length(sig_taus)
ccols = pcols(nsig_tau)
for i = 1:nsig_tau
    cirp = CIRProcess(mu_tau, sig_taus[i], theta_tau)
    ptau = pdf(ssdist(cirp), taus)
    add(p, Curve(taus, 0.5.+0.1*ptau, color=ccols[i], kind="dashed"))
end
pcurve = cell(nsig_tau)
for i = 1:nsig_tau
    pcurve[i] = Curve(taus, bound[i,:], color=ccols[i], width=pwidth)
    setattr(pcurve[i], label=@sprintf("%2.4f", sig_taus[i]))
    add(p, pcurve[i])
end
add(p, Legend(0.1, 0.9, pcurve))
display(p)
println("-- Figure saved as $figpath/dp2_examples_sig_tau.eps")
savefig(p, joinpath(figpath, "dp2_examples_sig_tau.eps"))


## Bound for different theta_tau
@readdata(datafile, "var_theta_tau", mu_tau, sig_tau, theta_taus, bound, taus)
p = FramedPlot(title = "Optimal bound for different \\tau  speeds",
    xlabel = "\\tau", ylabel = "g_\\theta(\\tau)",
    xrange = [taus[1], taus[div(length(taus),2)]], yrange = [0.5, 1.0],
    aspect_ratio = 3/4)
ntheta_tau = length(theta_taus)
ccols = pcols(ntheta_tau)
for i = 1:ntheta_tau
    cirp = CIRProcess(mu_tau, sig_tau, theta_taus[i])
    ptau = pdf(ssdist(cirp), taus)
    add(p, Curve(taus, 0.5.+0.1*ptau, color=ccols[i], kind="dashed"))
end
pcurve = cell(ntheta_tau)
for i = 1:ntheta_tau
    pcurve[i] = Curve(taus, bound[i,:], color=ccols[i], width=pwidth)
    setattr(pcurve[i], label=@sprintf("%4.2f", theta_taus[i]))
    add(p, pcurve[i])
end
add(p, Legend(0.1, 0.9, pcurve))
display(p)
println("-- Figure saved as $figpath/dp2_examples_theta_tau.eps")
savefig(p, joinpath(figpath, "dp2_examples_theta_tau.eps"))


## Bound for different ti's
@readdata(datafile, "var_ti", mu_tau, sig_tau, theta_tau, bound, taus, tis)
cirp = CIRProcess(mu_tau, sig_tau, theta_tau)
ptau = pdf(ssdist(cirp), taus)
p = FramedPlot(title = "Optimal bound for different t_i's",
    xlabel = "\\tau", ylabel = "g_\\theta(\\tau)",
    xrange = [taus[1], taus[div(length(taus),2)]], yrange = [0.5, 1.0],
    aspect_ratio = 3/4)
add(p,Curve(taus, 0.5.+0.2*ptau/maximum(ptau), kind="dashed"))
nti = length(tis)
ccols = pcols(nti)
pcurve = cell(nti);
for i = 1:nti
    pcurve[i] = Curve(taus, bound[i,:], color=ccols[i], width=pwidth)
    setattr(pcurve[i], label=@sprintf("%6.2f", tis[i]))
    add(p, pcurve[i])
end
add(p, Legend(0.1, 0.9, pcurve))
display(p)
println("-- Figure saved as $figpath/dp2_examples_ti.eps")
savefig(p, joinpath(figpath, "dp2_examples_ti.eps"))


## Bound for different tp's
@readdata(datafile, "var_tp", mu_tau, sig_tau, theta_tau, bound, taus, tps)
cirp = CIRProcess(mu_tau, sig_tau, theta_tau)
ptau = pdf(ssdist(cirp), taus)
p = FramedPlot(title = "Optimal bound for different t_p's",
    xlabel = "\\tau", ylabel = "g_\\theta(\\tau)",
    xrange = [taus[1], taus[div(length(taus),2)]], yrange = [0.5, 1.0],
    aspect_ratio = 3/4)
add(p,Curve(taus, 0.5.+0.2*ptau/maximum(ptau), kind="dashed"))
ntp = length(tps)
ccols = pcols(ntp)
pcurve = cell(ntp);
for i = 1:ntp
    pcurve[i] = Curve(taus, bound[i,:], color=ccols[i], width=pwidth)
    setattr(pcurve[i], label=@sprintf("%6.2f", tps[i]))
    add(p, pcurve[i])
end
add(p, Legend(0.1, 0.9, pcurve))
display(p)
println("-- Figure saved as $figpath/dp2_examples_tp.eps")
savefig(p, joinpath(figpath, "dp2_examples_tp.eps"))
