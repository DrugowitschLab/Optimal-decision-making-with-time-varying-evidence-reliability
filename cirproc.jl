# Module to perform simulations of the CIR process
# CIR = Cox-Ingersoll-Ross (1985)

module CIRProc

import Distributions: Gamma, Poisson, Chisq, rand, pdf, quantile
import Base: rand, mean, var

export CIRProcess, ssdist, scale, shape, rand, invmean, empmean,
    ExactCIRSim, EulerCIRSim, XCIRState,
    AbstractXCIRBound, XCIRTimeBound, XCIRParticleBound, XCIRXTauBound,
    AbstractXCIRSim, XCIRSim, XCIRDMSim, getdt, getmu, getcirp,
    FullXCIRTraj, StatsXCIRTraj, reset!, push!, first, last, lastt, mean, var,
    randtraj!, simpct, simproc


# ----------------------------------------------------------------------------
# CIR Process & Simulation
# ----------------------------------------------------------------------------

# -- Process --

immutable CIRProcess
    mu::Float64
    sig::Float64
    theta::Float64

    function CIRProcess(mu::Real, sig::Real, theta::Real)
        mu, sig, theta = float(mu), float(sig), float(theta)
        mu > 0 || error("mu needs to be positive")
        sig > 0 || error("sig needs to be positive")
        theta >= 0 || error("theta needs to be non-negative")
        new(mu, sig, theta)
    end

    CIRProcess() = CIRProcess(0.4, 0.2, 2)
end

# process parameters
shape(p::CIRProcess) = p.mu^2 / p.sig^2
scale(p::CIRProcess) = p.sig^2 / p.mu
pdf(p::CIRProcess, x::Real) = pdf(ssdist(p), x)
quantile(p::CIRProcess, x::Real) = quantile(ssdist(p), x)
# returns the steady state distribution
ssdist(p::CIRProcess) = Gamma(shape(p), scale(p))
# sample from steady state distribution
rand(p::CIRProcess) = rand(ssdist(p))
# mean of inverse steady state distribution
invmean(p::CIRProcess) = p.mu > p.sig ? p.mu / (p.mu * p.mu - p.sig * p.sig) : Inf

# for dgrat() and drcomp() in empmean
include("specialfuns.jl")

# empirical expectation of piece-wise linear function
#
# d is the process, x is the points at which the function is known, and fx is
# the function value at these points. x needs to be strictly increasing and
# cover all the parts of the steady-state distribution of p with mass.
#
# The function returns <f(x)>, x ~ ssdist(p)
function empmean{T1 <: Real, T2 <: Real}(p::CIRProcess, x::Vector{T1}, fx::Vector{T2})
    # computation based on assuming piece-wise linear fx and solving
    # expectation analytically over [x[j], x[j+1]] blocks
    res = 0.0
    const k, theta = shape(p), scale(p)
    dgratj, drcompj = dgrat(k, x[1] / theta)[1], drcomp(k, x[1] / theta)
    for j = 1:(length(x)-1)
        const dgratj1 = dgrat(k, x[j+1] / theta)[1]
        const drcompj1 = drcomp(k, x[j+1] / theta)
        res += (theta * (fx[j] - fx[j+1]) * (k * (dgratj - dgratj1) - drcompj + drcompj1) +
                (fx[j+1] * x[j] - fx[j] * x[j+1]) * (dgratj - dgratj1)) / (x[j+1] - x[j])
        dgratj, drcompj = dgratj1, drcompj1
    end
    res
end


# -- Simulations --
# simulations follow
# Anderson, Jaeckel, Kahl (2010): Simulation of Square-Root Processes
abstract AbstractCIRSim


# exact simulation
immutable ExactCIRSim <: AbstractCIRSim
    cirp::CIRProcess
    nt_dt::Float64
    d::Float64
    exp_theta_dt_nt_dt::Float64

    function ExactCIRSim(cirp::CIRProcess, dt::Float64)
        const exp_tmp = exp(-cirp.theta*dt)
        const nt_tmp = 2cirp.mu*exp_tmp/(cirp.sig^2*(1-exp_tmp))
        new(cirp, nt_tmp, 2cirp.mu^2/cirp.sig^2, exp_tmp/nt_tmp)
    end
end
rand(p::ExactCIRSim, curtau::Real) = p.exp_theta_dt_nt_dt *
    rand(Chisq(p.d + 2rand(Poisson(0.5curtau * p.nt_dt))))

# approximate simulation by truncated Euler method
immutable EulerCIRSim <: AbstractCIRSim
    cirp::CIRProcess
    theta_tau_dt::Float64
    eps_dt::Float64

    EulerCIRSim(cirp::CIRProcess, dt::Float64) =
        new(cirp, cirp.theta*dt, cirp.sig*sqrt(2cirp.theta*dt/cirp.mu))
end
function rand(p::EulerCIRSim, curtau::Float64)
    const taup = max(0.0, curtau)
    curtau + p.theta_tau_dt * (p.cirp.mu - taup) + 
        p.eps_dt * sqrt(taup) * randn()
end

# generates a sample path of the CIR process
#
# p is a tau process
# n is the number of steps of simulate
function simcir(p::AbstractCIRSim, n::Int)
    taus = fill(NaN, n)
    taus[1] = rand(p.cirp)
    for i = 2:n
        taus[i] = rand(p, taus[i-1])
    end
    taus
end


# ----------------------------------------------------------------------------
# XCIR Process Simulation
# ----------------------------------------------------------------------------

# -- State --

# state of XCIR process
immutable XCIRState
    x::Float64
    tau::Float64
    XCIRState(x::Real, tau::Real) = new(float64(x), float64(tau))
    XCIRState(p::CIRProcess) = XCIRState(0.0, rand(p))
    XCIRState() = XCIRState(0.0, 1.0)
end


# -- Bounds --

# bound on XCIR process state & time
abstract AbstractXCIRBound

# bound on time only
immutable XCIRTimeBound <: AbstractXCIRBound
    t::Float64

    function XCIRTimeBound(t::Real)
        t > zero(t) || error("t needs to be positive")
        new(float(t))
    end
end
crossedbound(b::XCIRTimeBound, xtau::XCIRState, t::Real) = float(t) >= b.t

# bound on belief/particle only
immutable XCIRParticleBound <: AbstractXCIRBound
    x::Float64

    function XCIRParticleBound(x::Real)
        x > zero(x) || error("x needs to be positive")
        new(float(x))
    end
end
crossedbound(b::XCIRParticleBound, xtau::XCIRState, t::Real) = abs(xtau.x) >= b.x

# bound on particle, depending on tau
#
# taux a vector, specifying for each tau the bound in x (and -x). The
# corresponding taus are tau0, tau0+dtau, tau0+2dtau, ...
immutable XCIRXTauBound <: AbstractXCIRBound
    taux::Vector{Float64}
    dtau::Float64
    mintau::Float64
    imax::Float64

    function XCIRXTauBound(taux::Vector{Float64}, dtau::Real, mintau::Real)
        dtau, mintau = float(dtau), float(mintau)
        dtau > 0 || error("dtau needs to be postitive")
        mintau >= 0 || error("mintau needs to be non-negative")
        new(taux, dtau, mintau, float(length(taux) - 1))
    end
end
function crossedbound(b::XCIRXTauBound, xtau::XCIRState, t::Real)
    const x, tau = xtau.x, xtau.tau
    const i = (tau - b.mintau) / b.dtau
    const btau = i <= 0.0     ? b.taux[1] :
                 i >= b.imax  ? b.taux[end] :
                 isinteger(i) ? b.taux[iround(i) + 1] : begin
            const ix = ifloor(i)
            i -= float(ix)
            (1.0 - i) * b.taux[ix+1] + i * b.taux[ix+2]
        end
    abs(x) >= btau
end


# -- Simulation --

# simulating XCIRP process
abstract AbstractXCIRSim
getdt(s::AbstractXCIRSim) = s.dt
getmu(s::AbstractXCIRSim) = s.mu
getcirp(s::AbstractXCIRSim) = s.cirp

# simulating XCIRProcess with optimal (weighted) evidence accumulation
immutable XCIRSim <: AbstractXCIRSim
    cirp::CIRProcess
    mu::Float64
    dt::Float64
    cirs::AbstractCIRSim
    dmu::Float64

    function XCIRSim(cirp::CIRProcess, mu::Real, dt::Real, cirs::AbstractCIRSim)
        dt > zero(dt) || error("dt needs to be positive")
        new(cirp, float(mu), float(dt), cirs, float(mu * dt))
    end
    XCIRSim(cirp, mu, dt) = XCIRSim(cirp, mu, dt, EulerCIRSim(cirp, dt))
    XCIRSim(cirp, dt) = XCIRSim(cirp, 1.0, dt, EulerCIRSim(cirp, dt))
end
# draws a sample given current (x, tau) state
function rand(p::XCIRSim, curxtau::XCIRState)
    const tau = rand(p.cirs, curxtau.tau)
    const taupos = max(0.0, tau)
    XCIRState(curxtau.x + taupos * p.dmu + sqrt(p.dt * taupos) * randn(), tau)
end

# simulating unweighted XCIR process, diffusion model-style
immutable XCIRDMSim <: AbstractXCIRSim
    cirp::CIRProcess
    mu::Float64
    dt::Float64
    cirs::AbstractCIRSim
    dmu::Float64

    function XCIRDMSim(cirp::CIRProcess, mu::Real, dt::Real, cirs::AbstractCIRSim)
        dt > zero(dt) || error("dt needs to be positive")
        new(cirp, float(mu), float(dt), cirs, float(mu * dt))
    end
    XCIRDMSim(cirp, mu, dt) = XCIRDMSim(cirp, mu, dt, EulerCIRSim(cirp, dt))
    XCIRDMSim(cirp, dt) = XCIRDMSim(cirp, 1.0, dt, EulerCIRSim(cirp, dt))
end
# draws a sample given current (x, tau) state
function rand(p::XCIRDMSim, curxtau::XCIRState)
    const tau = rand(p.cirs, curxtau.tau)
    XCIRState(curxtau.x + p.dmu + sqrt(p.dt / max(eps(), tau)) * randn(), tau)
end


# -- State Trajectory --

# sample trajectory of XCIR process - maintains statistics about trajectory
abstract AbstractXCIRTraj

# stores full trajectory of XCIR process
type FullXCIRTraj <: AbstractXCIRTraj
    dt::Float64
    n::Int
    nincr::Int
    xtaus::Matrix{Float64}

    function FullXCIRTraj(xtau0::XCIRState, dt::Real)
        dt = float(dt)
        dt > 0.0 || error("dt needs to be positive")
        nincr = max(1, int(div(1, dt)))
        xtaus = fill(NaN, nincr, 2)
        xtaus[1,1] = xtau0.x
        xtaus[1,2] = xtau0.tau
        new(dt, 1, nincr, xtaus)
    end
    FullXCIRTraj(dt) = FullXCIRTraj(XCIRState(), dt)
end
function reset!(t::FullXCIRTraj, xtau0::XCIRState)
    t.xtaus = fill(NaN, t.nincr, 2)
    t.xtaus[1,1] = xtau0.x
    t.xtaus[1,2] = xtau0.tau
    t.n = 1;
end
reset!(t::FullXCIRTraj, p::CIRProcess) = reset!(t, rand(p))
function push!(t::FullXCIRTraj, xtau::XCIRState)
    if t.n == size(t.xtaus, 1)
        t.xtaus = [t.xtaus, fill(NaN, t.nincr, 2)]
    end
    t.n += 1
    t.xtaus[t.n,1] = xtau.x
    t.xtaus[t.n,2] = xtau.tau;
end
first(t::FullXCIRTraj) = XCIRState(t.xtaus[1,1], t.xtaus[1,2])
last(t::FullXCIRTraj) = XCIRState(t.xtaus[t.n,1], t.xtaus[t.n,2])
lastt(t::FullXCIRTraj) = (t.n - 1) * t.dt
mean(t::FullXCIRTraj) = XCIRState(mean(t.xtaus[1:t.n,1]), mean(t.xtaus[1:t.n,2]))
var(t::FullXCIRTraj) = XCIRState(var(t.xtaus[1:t.n,1]), var(t.xtaus[1:t.n,2]))

# only maintains trajectory statistics of XCIR process
type StatsXCIRTraj <: AbstractXCIRTraj
    dt::Float64
    n::Int
    x0::Float64
    tau0::Float64
    xn::Float64
    taun::Float64
    xsum::Float64
    tausum::Float64
    x2sum::Float64
    tau2sum::Float64

    function StatsXCIRTraj(xtau0::XCIRState, dt::Real)
        dt > zero(dt) || error("dt needs to be positive")
        new(float(dt), 1, xtau0.x, xtau0.tau, xtau0.x, xtau0.tau,
            xtau0.x, xtau0.tau, xtau0.x * xtau0.x, xtau0.tau * xtau0.tau)
    end
    StatsXCIRTraj(dt) = StatsXCIRTraj(XCIRState(), dt)
end
function reset!(t::StatsXCIRTraj, xtau0::XCIRState)
    t.x0, t.tau0, t.xn, t.taun = xtau0.x, xtau0.tau, xtau0.x, xtau0.tau
    t.xsum, t.x2sum = xtau0.x, xtau0.x * xtau0.x
    t.tausum, t.tau2sum = xtau0.tau, xtau0.tau * xtau0.tau
    t.n = 1;
end
reset!(t::StatsXCIRTraj, p::CIRProcess) = reset!(t, rand(p))
function push!(t::StatsXCIRTraj, xtau::XCIRState)
    t.xn, t.taun = xtau.x, xtau.tau
    t.xsum += xtau.x
    t.tausum += xtau.tau
    t.x2sum += xtau.x * xtau.x
    t.tau2sum += xtau.tau * xtau.tau
    t.n += 1;
end
first(t::StatsXCIRTraj) = XCIRState(t.x0, t.tau0)
last(t::StatsXCIRTraj) = XCIRState(t.xn, t.taun)
lastt(t::StatsXCIRTraj) = (t.n - 1) * t.dt
mean(t::StatsXCIRTraj) = XCIRState(t.xsum / t.n, t.tausum / t.n)
var(t::StatsXCIRTraj) =
    XCIRState(t.x2sum / t.n - (t.xsum / t.n)^2, t.tau2sum / t.n - (t.tausum / t.n)^2)

# -- Bounded Simulations --

# simulates a trajectory until the bound is reached
# 
# The stimulation starts at last(t) and stops when the bound is crossed.
function randtraj!(t::AbstractXCIRTraj, s::AbstractXCIRSim, b::AbstractXCIRBound)
    xtau = last(t)
    while !crossedbound(b, xtau, lastt(t))
        xtau = rand(s, xtau)
        push!(t, xtau)
    end
end

# simulates process until bound is reached, returning some process state
#
# s is the simulator, b is the bound.
#
# The function returns (xtau, t) where xtau is the XCIRState and t is the time
# after bound crossing.
function rand(s::AbstractXCIRSim, b::AbstractXCIRBound)
    const dt = getdt(s)
    xtau = XCIRState(0.0, rand(getcirp(s)))
    t = 0.0
    while !crossedbound(b, xtau, t)
        xtau = rand(s, xtau)
        t += dt
    end
    xtau, t
end

# returns p(correct) and bound-crossing time for single trial
#
# s is the simulator, b is the bound
#
# The function returns (pc, t) where pc is the probability correct, and t is
# the bound-crossing time. pc is 1.0 if the sign of s.mu is the same as the
# sign of the final x.
function simpct(s::AbstractXCIRSim, b::AbstractXCIRBound)
    const xtau, t = rand(s, b)
    const pc = getmu(s) >= 0.0 ? (xtau.x >= 0.0 ? 1.0 : 0.0) :
                                 (xtau.x < 0.0 ? 1.0 : 0.0)
    pc, t
end
# Analytical solutions for optimal (weighted) simulation and bound on particle
function simpct(s::XCIRSim, b::XCIRParticleBound)
    const thetamin = 1e-10
    const pc = 1.0 / (1.0 + exp(-2.0b.x))
    # for small theta we can use the analytical solution
    const cirp = getcirp(s)
    if cirp.theta < thetamin
        return pc, invmean(cirp) * b.x * tanh(b.x)
    else
        const xtau, t = rand(s, b)
        return pc, t
    end
end


# simulates a set of bounded trials and returns statistics
#
# s is the XCIR simulator, b is the bound, and n is the number of trials.
#
# The function returns a dictionary with the following elements:
# - rt: vector of decision times per trial
# - choice: vector of choices, 1.0 for upper boundary, 0.0 for lower
# - tau0: vector of initial tau's
# - taun: vector of tau's at decision time
# - avgtau: vector of average tau's
function simproc(s::AbstractXCIRSim, b::AbstractXCIRBound, n::Int)
    const cirp = getcirp(s)
    t = StatsXCIRTraj(getdt(s))
    # simulate trajectories
    rt, choice = Array(Float64, n), Array(Float64, n)
    tau0, taun, avgtau = Array(Float64, n), Array(Float64, n), Array(Float64, n)
    for i = 1:n
        reset!(t, XCIRState(0.0, rand(cirp)))
        randtraj!(t, s, b)
        rt[i] = lastt(t)
        choice[i] = last(t).x >= 0.0  ? 1.0 : 0.0
        tau0[i], taun[i], avgtau[i] = first(t).tau, last(t).tau, mean(t).tau
    end
    ["rt" => rt, "choice" => choice,
     "tau0" => tau0, "taun" => taun, "avgtau" => avgtau]
end


end # module