# Script simulating the CIR process with different parameters,
# and writing the result to cir_examples.jld

using CIRProc

include("utils.jl")

# general settings
datafile = "cir_examples.jld"

#maxtau_prctile = 0.99

# process base parameters
mu_tau = 0.4
sig_tau = 0.2
theta_tau = 2
dt = 0.005
tmax = 10
n = int(div(tmax, dt))
ts = [(0:(n-1)) * dt]

## different mu_tau
println("-- Simulations for different mu_tau")
mu_taus = [sqrt(0.2^2/2), 0.40, 1.00]
nmu_tau = length(mu_taus)
tau_traj = fill(NaN, nmu_tau, n)
for i in 1:nmu_tau
    p = CIRProcess(mu_taus[i], sig_tau, theta_tau)
    tau_traj[i,:] = CIRProc.simcir(ExactCIRSim(p, dt), n)
end
println("-- Writing data to $datafile/var_mu_tau")
@writedata(datafile, "var_mu_tau",
    mu_taus, sig_tau, theta_tau, ts, tau_traj)
println()


## different sig_tau
println("-- Simulations for different sig_tau")
sig_taus = [0.05, 0.1, 0.2, sqrt(2*mu_tau^2)] # needs to be smaller sqrt(2 mu_tau^2)
nsig_tau = length(sig_taus)
tau_traj = fill(NaN, nsig_tau, n)
for i in 1:nsig_tau
    p = CIRProcess(mu_tau, sig_taus[i], theta_tau)
    tau_traj[i,:] = CIRProc.simcir(ExactCIRSim(p, dt), n)
end
println("-- Writing data to $datafile/var_mu_tau")
@writedata(datafile, "var_sig_tau",
    mu_tau, sig_taus, theta_tau, ts, tau_traj)
println()


## theta_tau
println("-- Simulations for different theta_tau")
theta_taus = [0.25, 1.0, 4.0, 16.0]
ntheta_tau = length(theta_taus)
tau_traj = fill(NaN, ntheta_tau, n)
for i in 1:ntheta_tau
    p = CIRProcess(mu_tau, sig_tau, theta_taus[i])
    tau_traj[i,:] = CIRProc.simcir(ExactCIRSim(p, dt), n)
end
println("-- Writing data to $datafile/var_theta_tau")
@writedata(datafile, "var_theta_tau",
    mu_tau, sig_tau, theta_taus, ts, tau_traj)
println()

