# Script running standard dp examples to find the bound for different
# task and process parameters
#
# All data is saved to dp2_examples.jld and can be plotted with
# dp2_examples_plot.jl
#
# In constrast to dp_examples.jl, this script uses the corrected VI2 module.

# @everywhere, as later called in @parallel
@everywhere using VI2, CIRProc

include("utils.jl")

# general settings
datafile = "dp2_examples.jld"
maxtau_prctile = 0.99

# process/task parameters
mu_tau = 0.4
sig_tau = 0.2
theta_tau = 2
rcorr, rincorr, c = 1, 0, 0.1
ti, tp = 1, 0

# PDE solver settings
testing = false
if testing
    dt, ng, ntau = 0.01, 100, 100
else
    dt, ng, ntau = 0.005, 500, 500
end


## Bounds for different costs
println("-- Optimal bounds for different costs")
cs = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]
nc = length(cs)
p = CIRProcess(mu_tau, sig_tau, theta_tau)
maxtau = 2quantile(p, maxtau_prctile)
r = GTauProcSpace(p, ng, ntau, maxtau)
res = @parallel vcat for i in 1:nc
    c1 = cs[i]
    println("c = $c1")
    t = FDTask(p, c1, rcorr, rincorr)
    vi(t, r, dt)
end
bound = fill(NaN, nc, ntau)
v = fill(NaN, nc, ng, ntau)
er = fill(NaN, nc)
for i in 1:nc
    bound[i,:], v[i,:,:], er[i] = res[i]
end
gs = getgs(ng)
taus = gettaus(ntau, maxtau)
println("-- Writing data to $datafile/var_c")
@writedata(datafile, "var_c",
    mu_tau, sig_tau, theta_tau, rcorr, rincorr, cs, bound, v, er, gs, taus)
println()


## Bounds for different mu_tau
println("-- Optimal bounds for different mu_tau")
mu_taus = [sqrt(0.2^2/2), 0.40, 1.00]
nmu_tau = length(mu_taus)
maxtau = 2quantile(
    CIRProcess(maximum(mu_taus), sig_tau, theta_tau), maxtau_prctile)
res = @parallel vcat for i in 1:nmu_tau
    mu_tau1 = mu_taus[i]
    println("mu_tau = $mu_tau1")
    p = CIRProcess(mu_tau1, sig_tau, theta_tau)
    t = FDTask(p, c, rcorr, rincorr)
    r = GTauProcSpace(p, ng, ntau, maxtau)
    vi(t, r, dt)
end
bound = fill(NaN, nmu_tau, ntau)
v = fill(NaN, nmu_tau, ng, ntau)
er = fill(NaN, nmu_tau)
for i in 1:nmu_tau
    bound[i,:], v[i,:,:], er[i] = res[i]
end
gs = getgs(ng)
taus = gettaus(ntau, maxtau)
println("--Writing data to $datafile/var_mu_tau")
@writedata(datafile, "var_mu_tau",
    mu_taus, sig_tau, theta_tau, rcorr, rincorr, c, bound, v, er, gs, taus)
println()


## Bounds for different sig_tau
println("-- Optimal bounds for different sig_tau")
sig_taus = [0.05, 0.1, 0.2, sqrt(2*mu_tau^2)] # needs to be smaller sqrt(2 mu_tau^2)
nsig_tau = length(sig_taus)
maxtau = 2quantile(
    CIRProcess(mu_tau, maximum(sig_taus), theta_tau), maxtau_prctile)
res = @parallel vcat for i in 1:nsig_tau
    sig_tau1 = sig_taus[i]
    println("sig_tau = $sig_tau1")
    p = CIRProcess(mu_tau, sig_tau1, theta_tau)
    r = GTauProcSpace(p, ng, ntau, maxtau)
    t = FDTask(p, c, rcorr, rincorr)
    vi(t, r, dt)
end
bound = fill(NaN, nsig_tau, ntau)
v = fill(NaN, nsig_tau, ng, ntau)
er = fill(NaN, nsig_tau)
for i = 1:nsig_tau
    bound[i,:], v[i,:,:], er[i] = res[i]
end
gs = getgs(ng)
taus = gettaus(ntau, maxtau)
println("--Writing data to $datafile/var_mu_tau")
@writedata(datafile, "var_sig_tau",
    mu_tau, sig_taus, theta_tau, rcorr, rincorr, c, bound, v, er, gs, taus)
println()


## Bounds for different theta_tau
println("-- Optimal bounds for different theta_tau")
theta_taus = [0.25, 1.0, 4.0, 16.0]
ntheta_tau = length(theta_taus)
maxtau = 2quantile(
    CIRProcess(mu_tau, sig_tau, maximum(theta_taus)), maxtau_prctile)
res = @parallel vcat for i in 1:ntheta_tau
    theta_tau1 = theta_taus[i]
    println("theta_tau = $theta_tau1")
    p = CIRProcess(mu_tau, sig_tau, theta_tau1)
    t = FDTask(p, c, rcorr, rincorr)
    r = GTauProcSpace(p, ng, ntau, maxtau)
    vi(t, r, dt)
end
bound = fill(NaN, ntheta_tau, ntau)
v = fill(NaN, ntheta_tau, ng, ntau)
er = fill(NaN, ntheta_tau)
for i = 1:ntheta_tau
    bound[i,:], v[i,:,:], er[i] = res[i]
end
gs = getgs(ng)
taus = gettaus(ntau, maxtau)
println("--Writing data to $datafile/var_theta_tau")
@writedata(datafile, "var_theta_tau",
    mu_tau, sig_tau, theta_taus, rcorr, rincorr, c, bound, v, er, gs, taus)
println()


## Bounds for different inter-trial intervals
println("-- Optimal bounds for different inter-trial intervals")
tis = [0.04 0.2 1 5 25 125 625]
nti = length(tis)
p = CIRProcess(mu_tau, sig_tau, theta_tau)
maxtau = 2quantile(p, maxtau_prctile)
r = GTauProcSpace(p, ng, ntau, maxtau)
res = @parallel vcat for i in 1:nti
    ti1 = tis[i]
    println("ti = $ti1")
    t = RTTask(p, c, rcorr, rincorr, ti1, tp)
    vi(t, r, dt)
end
bound = fill(NaN, nti, ntau)
v = fill(NaN, nti, ng, ntau)
rr = fill(NaN, nti)
for i in 1:nti
    bound[i,:], v[i,:,:], rr[i] = res[i]
end
gs = getgs(ng)
taus = gettaus(ntau, maxtau)
println("-- Writing data to $datafile/var_ti")
@writedata(datafile, "var_ti",
    mu_tau, sig_tau, theta_tau, rcorr, rincorr, c, tis, tp, bound, v, rr, gs, taus)
println()


## Bounds for different penalty times
println("-- Optimal bounds for different penalty times")
tps = [0 0.1 0.3 0.9 2.7 8.1 24.3]
ntp = length(tps)
p = CIRProcess(mu_tau, sig_tau, theta_tau)
maxtau = 2quantile(p, maxtau_prctile)
r = GTauProcSpace(p, ng, ntau, maxtau)
res = @parallel vcat for i in 1:ntp
    tp1 = tps[i]
    println("tp = $tp1")
    t = RTTask(p, c, rcorr, rincorr, ti, tp1)
    vi(t, r, dt)
end
bound = fill(NaN, ntp, ntau)
v = fill(NaN, ntp, ng, ntau)
rr = fill(NaN, ntp)
for i in 1:ntp
    bound[i,:], v[i,:,:], rr[i] = res[i]
end
gs = getgs(ng)
taus = gettaus(ntau, maxtau)
println("-- Writing data to $datafile/var_tp")
@writedata(datafile, "var_tp",
    mu_tau, sig_tau, theta_tau, rcorr, rincorr, c, ti, tps, bound, v, rr, gs, taus)
println()


