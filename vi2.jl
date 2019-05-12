# Script performing value iteration with tau process
#
# This is a fixed / extended version of the VI module with correct(ed) boundary
# conditions

module VI2

import Roots: fzero, ConvergenceFailed
import Distributions: Gamma, pdf, quantile
import Base: size
using NLopt

using CIRProc

export FDTask, RTTask, getrr,  vi, gettaus, getgs, pdf,
    max_rr_particle,
    XTauProcSpace, GTauProcSpace

# discretised parameter space for VI
abstract AbstractProcSpace
# PDE solver to compute the expectation
abstract AbstractPDE2DSolver

# ----------------------------------------------------------------------------
# Task definition
# ----------------------------------------------------------------------------

abstract AbstractTask

# fixed duration task (infinite duration)
immutable FDTask <: AbstractTask
    p::CIRProcess
    c::Float64
    rcorr::Float64
    rincorr::Float64

    function FDTask(p::CIRProcess, c::Real, rcorr::Real, rincorr::Real)
        new(p, float(c), float(rcorr), float(rincorr))
    end
    FDTask(p, c) = FDTask(p, c, 1, 0)
    FDTask(p) = FDTask(p, 0.1)
end
getrr(t::FDTask, pc::Real, dt::Real) =
    float(pc) * t.rcorr + (1.0 - float(pc)) * t.rincorr - float(dt) * t.c

# reaction time task
immutable RTTask <: AbstractTask
    p::CIRProcess
    c::Float64
    rcorr::Float64
    rincorr::Float64
    ti::Float64
    tp::Float64

    function RTTask(p::CIRProcess,
                    c::Real, rcorr::Real, rincorr::Real, ti::Real, tp::Real)
        ti, tp = float(ti), float(tp)
        ti >= 0 || error("ti needs to be non-negative")
        tp >= 0 || error("tp needs to be non-negative")
        new(p, float(c), float(rcorr), float(rincorr), ti, tp)
    end
    RTTask(p, c, ti) = RTTask(p, c, 1, 0, ti, 0)
    RTTask(p, c) = RTTask(p, c, 1)
    RTTask(p) = RTTask(p)
end
getrr(t::RTTask, pc::Real, dt::Real) =
    (float(pc) * t.rcorr + (1.0 - float(pc)) * t.rincorr - float(dt) * t.c) /
    (dt + t.ti + (1.0 - float(pc)) * t.tp)


# ----------------------------------------------------------------------------
# Heuristic bound optimization
# ----------------------------------------------------------------------------

include("rr_optim.jl")


# ----------------------------------------------------------------------------
# Dynamic programming
# ----------------------------------------------------------------------------


# Performs value iteration to find the optimal bound
#
# t is the task description
# r is the nx x ny discretized process parameter space
# dt is the time discretisation
#
# The function returns (bound, v, er), where bound is an nx-element
# array, v is the nx x ny value function, and er is the expected reward
function vi(t::FDTask, r::AbstractProcSpace, dt::Float64)
    rdiff = t.rcorr - t.rincorr
    tnorm = FDTask(t.p, t.c / rdiff)
    bound, v = vinorm(t, r, dt)
    v = rdiff * v .+ t.rincorr
    er = avgreward(v, gettaus(r), t.p)
    return bound, v, er
end


# Performs value iteration to find the optimal bound
#
# The function assumes 
#
# t is the task description
# r is the nx x ny discretized process parameter space
# dt is the time discretisation
#
# The function returns (bound, v, rho), where bound is an ny-element
# array, v is the nx x ny value function, and rho is the reward rate
function vi(t::RTTask, r::AbstractProcSpace, dt::Float64; verbose::Bool=false)
    const maxiter, roottol = 20, 1e-12
    rdiff = t.rcorr - t.rincorr
    bestrho, bestv0 = 0.0, Inf

    function rr_initial_value(rho::Real, rc::AbstractProcSpace)
        k = rdiff + rho * t.tp
        tnorm = FDTask(t.p, (t.c + rho) / k)
        bound, v = vinorm(tnorm, rc, dt)
        v0 = avgreward(v * k .+ t.rincorr .- rho * (t.ti + t.tp), gettaus(rc), t.p)
        !verbose || println("rho = $rho, <v(1/2,tau)> = $v0")
        if abs(v0) < bestv0; bestrho, bestv0 = rho, abs(v0); end
        v0
    end

    # reward rate bounds, for DT from 0 to Inf, R from rincorr to rcorr
    rrs = [t.rcorr / t.ti, (t.rcorr + t.rincorr) / (2 * t.ti + t.tp), -t.c]
    rrmax, rrmin = maximum(rrs), minimum(rrs)
    rrini = rrmin + 0.8 * (rrmax - rrmin)

    # find reward rate that makes v(1/2) = 0
    const nx, ny = size(r)
    const xdiv, ydiv = 5, 5
    rcoarse = coarser(r, xdiv, ydiv)
    const nxcoarse, nycoarse = size(rcoarse)
    !verbose || println("-- Root finding to find reward rate")
    !verbose || println("-- Coarse grid, nx = $nxcoarse, ny = $nycoarse")
    # make sure to return last rho, even if convergence failed
    rho = try
        fzero(x -> rr_initial_value(x, rcoarse),
              rrini, [rrmin, rrmax], max_iter=maxiter, tol=roottol)
    catch y
        if isa(y, ConvergenceFailed)
            bestrho
        else
            rethrow(y)
        end
    end
    !verbose || println("-- Fine grid, nx = ", nx, ", ny = ", ny)
    bestv0 = Inf
    rho = try
        fzero(x -> rr_initial_value(x, r),
              rho, rho .+ 0.1 * (rrmax - rrmin) * [-1, 1],
              max_iter=maxiter, tol=roottol)
    catch y
        if isa(y, ConvergenceFailed)
            bestrho
        else
            rethrow(y)
        end
    end
    !verbose || println("-- Reward rate rho = $rho")

    k = rdiff + rho * t.tp;
    tnorm = FDTask(t.p, (t.c + rho) / k)
    bound, v = vinorm(tnorm, r, dt)

    return bound, v * k .+ t.rincorr .- rho * (t.ti + t.tp), rho
end


# returns the tau discretisation
#
# ntau is the number of tau steps
# maxtau is the largest tau to represent
# t is the task, in which case tau is discretised up to 2 * 99th prctile
#
# The function returns a vector of size ntau
gettaus(ntau::Int, maxtau::Float64) = linspace(0.0, maxtau, ntau)
gettaus(ntau::Int, p::CIRProcess) = gettaus(ntau, 2*quantile(p, 0.99))


# returns the g discretisation
#
# ng is the number of g steps
#
# The function returns a vector of size ng
getgs(ng::Int) = linspace(0.0, 1.0, ng)


# performs value iteration for fixed duration tasks and normalised costs
#
# t is the task description. Only the process and cost are considered
# r is the nx x ny discretized process parameter space
# dt is the time discretisation
#
# The function returns (v, bound), where v is the nx x ny final value function,
# and bound is the bound found over ny-space.
function vinorm(t::FDTask, r::AbstractProcSpace, dt::Float64)
    p = PDE2DADISolver(r, dt)
    vini, vd, gs = getvini(r), getvd(r), getgs(r)
    vinormiter(p, vini, vd, gs, t.c, dt)
end


# returns value function and policy resulting from value iteration
#
# p is the solver to find the expected value function
# vini is the K x J matrix specifying the initial value function
# vd is the K x J matrix speficying the value function for immediate deicsions
# gs is the K-vector of beliefs
# c is the cost per unit time
# dt is the time discretisation (needs to match p)
# maxiter specifies the maximum number of VI steps
# maxvdiff is the largest acceptable abs(v - vprev).dt to consider convergence
#
# The function returns (v, bound), where v is the K x J final value function,
# and bound is the found bound over J-space. This bound is computed by
# interpolating over values in gs. It is, in fact, the only point where gs is
# being used.
#
# vini and vd need to correspond to each other at the boundaries in k, but
# not in j.
function vinormiter{T <: Real}(p::AbstractPDE2DSolver,
    vini::Matrix{T}, vd::Matrix{T}, gs::Vector{T}, c::Real, dt::Real,
    maxiter::Int=0, maxvdiff::Real=0.001)

    const K, J = size(vini, 1), size(vini, 2)
    @assert K == size(vd, 1) && J == size(vd, 2)
    @assert K == length(gs)
    @assert dt > zero(dt)
    if maxiter == 0
        maxiter = iceil(50.0/dt)
    end

    const dc = c * dt
    converged = false
    v = vini
    vnext = Array(Float64, K, J)
    for n = 2:maxiter
        vprev = copy(v)
        vnext = expectedu(p, v)

        # bounds except for j = 1 need to remain unchanged
        @assert all([isapprox(vnext[k,end], vprev[k,end]) for k = 1:K])
        @assert all([isapprox(vnext[1,j], vprev[1,j]) for j = 1:J])
        @assert all([isapprox(vnext[end,j], vprev[end,j]) for j = 1:J])
        # boundaries for k = 1 and k = K need to be decision boundaries
        @assert all([isapprox(vnext[1,j], vd[1,j]) for j = 1:(J-1)])
        @assert all([isapprox(vnext[end,j], vd[end,j]) for j=  1:(J-1)])

        # based on vnext, update value function, using unrolled
        # v = max(vnext .- dc, vd). v[:,end] is unchanged to preserve
        # boundary condition at j=J
        for j = 1:(J-1), k = 1:K
            v[k,j] = max(vnext[k,j] - dc, vd[k,j])    
        end

        # bounds except for j = 1 need to remain unchanged
        @assert all([isapprox(v[k,end], vprev[k,end]) for k = 1:K])
        @assert all([isapprox(v[1,j], vprev[1,j]) for j = 1:J])
        @assert all([isapprox(v[end,j], vprev[end,j]) for j = 1:J])

        # compute value function difference
        if maxvdiff > zero(maxvdiff)
            vdiff = -Inf
            for j=1:J, k=1:K
                const vdiffjk = abs(v[k,j] - vprev[k,j])
                if vdiff < vdiffjk; vdiff = vdiffjk; end
            end
            if vdiff/dt <= maxvdiff; converged = true;  break;  end
        end
    end
    if maxvdiff > zero(maxvdiff) && !converged
        warn("value iteration did not converge within $maxiter iterations")
    end

    interpolatebound(vnext .- dc, vd, gs), v
end


# Returns the decision value function for (1,0) reward
#
# gs is ng-element array of reals
# 
# function returns ng x n array of the decision value function
function getvd(gs::Vector, n::Int)
    repmat(max(gs, 1.-gs), 1, n)
end


# Returns initial value function proposal over g's for given bound
#
# bound defines the bound location
# gs is ng-element array of reals
#
# function returns initial value function over g's that corresponds to bound
# at given location.
function getvini{T <: Real}(bound::T, gs::Vector{T})
    vini = 1.0/(2*bound-1)^3 * (bound^3*(3*bound-2) .-
        (1+6*bound*(bound-1))*gs .+ 6*bound*(bound-1)*gs.^2 .+ 2*gs.^3 .- gs.^4)
    max(vini, max(gs, 1.-gs))
end


# vnext is ng x n_nu array
# vd is ng x n_nu array
# gs is ng-element array of reals
#
# function returns bound on intersecting vnext and vd, using linear interp.
#
# the returned bound is a vector is size n_nu
function interpolatebound{T <: Real}(vnext::Matrix{T}, vd::Matrix{T}, gs::Vector{T})
    # argument properties
    ng, n_nu = size(vnext)
    dg = gs[2] - gs[1]
    
    # compute bound for each nu separately
    bound = fill(NaN, n_nu)
    g_lo = div(ng,2)+1
    for n = 1:n_nu
        g_idx = find(vd[g_lo:end,n] .>= vnext[g_lo:end,n])
        if isempty(g_idx)
            # vd >= vnext nowhere -> bound is at g = 1
            bound[n] = 1.0
        elseif g_idx[1] == 1
            # vd >= vnext everywhere -> bound is at g = 0.5
            bound[n] = 0.5
        else
            # adjust g_idx to be gs index before/below boundary
            g_idx = g_idx[1]+g_lo-2
            # find boundary by linear interpolation between g_idx and g_idx+1
            bound[n] = gs[g_idx] + dg/(
                1+(vd[g_idx+1,n]-vnext[g_idx+1,n])/(vnext[g_idx,n]-vd[g_idx,n]))
        end
    end
    
    bound
end


# Returns the expected reward at trial start
#
# The expected return is V(g,tau) at g=1/2 with the expectation over tau,
# assuming that tau follows the prior p(tau).
#
# v is the ng x ntau value function
# taus is the tau discretisation
# p are the CIR process parameters
#
# The function returns the scalar expected reward.
function avgreward{T <: Real}(v::Matrix{T}, taus::Vector{T}, p::CIRProcess) 
    # argument properties
    ng, ntau = size(v)
    @assert length(taus) == ntau

    # find values at belief g=0.5
    vg2 = fill(NaN, ntau)
    if mod(ng,2) == 0
        g2_idx = div(ng,2)
        vg2 = squeeze(mean(v[g2_idx:(g2_idx+1),:],1),1)
    else
        vg2 = v[div(ng,2)+1,:]
    end

    empmean(p, taus, vg2)
end


# ----------------------------------------------------------------------------
# Process spaces
# ----------------------------------------------------------------------------

# Process space over (X, tau)
immutable XTauProcSpace <: AbstractProcSpace
    xs::Vector{Float64}
    gs::Vector{Float64}
    taus::Vector{Float64}
    p::CIRProcess

    function XTauProcSpace(p::CIRProcess, nx::Int, ntau::Int, maxtau::Real)
        maxtau > zero(maxtau) || error("maxtau needs to be positive")
        const xrange = [-1.0, 1] * 0.5log(2.0nx - 1.0) # 1/2nx distance from [0,1]
        const xs = linspace(xrange[1], xrange[2], nx)
        new(xs, 1.0 ./ (1.0 .+ exp(-2.0xs)), gettaus(ntau, maxtau), p)
    end
end
XTauProcSpace(p::CIRProcess, nx::Int, ntau::Int) =
    XTauProcSpace(p, nx, ntau, 2.0quantile(p, 0.99))
size(r::XTauProcSpace) = length(r.xs), length(r.taus)
getgs(r::XTauProcSpace) = r.gs
gettaus(r::XTauProcSpace) = r.taus
getvd(r::XTauProcSpace) = getvd(r.gs, length(r.taus))
function getvini(r::XTauProcSpace)
    const nx, ntau = size(r)
    const gmax = max(r.gs[1], r.gs[end])
    vini = Array(Float64, nx, ntau)
    for j = 1:ntau
        # upper rim of v for large |x| and tau is at gmax
        vbase = getvini(0.5+(gmax-0.5)*(j-1)/(ntau-1), r.gs)
        vini[:,j] = (vbase .- gmax)*(ntau-j)/(ntau-1) .+ gmax;
    end
    vini
end
function moments(r::XTauProcSpace, dt::Real)
    const nx, ntau = size(r)
    const dx = (2.0r.gs .- 1.0) .* r.taus'             # (2g(X)-1) tau
    const dx2 = repmat(r.taus', nx, 1)                 # tau
    const dtau = repmat(r.p.theta * (r.p.mu .- r.taus'), nx, 1) # theta (mu-tau)
    const dtau2 = repmat(2.0r.p.theta * r.p.sig^2 / r.p.mu * r.taus',
                         nx, 1)                        # 2 theta sig^2 / mu tau
    dx, dx2, dtau, dtau2, zeros(nx, ntau), r.xs[2]-r.xs[1], r.taus[2]-r.taus[1]
end
coarser(r::XTauProcSpace, xdiv::Real, ydiv::Real) = 
    XTauProcSpace(r.p, int(div(length(r.xs),xdiv)), int(div(length(r.taus),ydiv)),
                  maximum(r.taus))


# Process space over (g, tau)
immutable GTauProcSpace <: AbstractProcSpace
    gs::Vector{Float64}
    taus::Vector{Float64}
    p::CIRProcess

    function GTauProcSpace(p::CIRProcess, ng::Int, ntau::Int, maxtau::Real)
        maxtau > zero(maxtau) || error("maxtau needs to be positive")
        new(linspace(0.0, 1.0, ng), gettaus(ntau, maxtau), p)
    end
end
GTauProcSpace(p::CIRProcess, ng::Int, ntau::Int) = 
    GTauProcSpace(p, ng, ntau, 2.0quantile(p, 0.99))
size(r::GTauProcSpace) = length(r.gs), length(r.taus)
getgs(r::GTauProcSpace) = r.gs
gettaus(r::GTauProcSpace) = r.taus
getvd(r::GTauProcSpace) = getvd(r.gs, length(r.taus))
function getvini(r::GTauProcSpace)
    const ng, ntau = size(r)    
    vini = Array(Float64, ng, ntau)
    for j = 1:ntau
        vbase = getvini(0.5+0.5*(j-1)/(ntau-1), r.gs)
        vini[:,j] = (vbase .- 1.0)*(ntau-j)/(ntau-1) .+ 1.0;
    end
    vini
end
function moments(r::GTauProcSpace, dt::Real)
    const ng, ntau = size(r)
    const dg2 = (4.0(1.0 .- r.gs).^2 .* r.gs.^2) .* r.taus' # 4(1-g)^2 g^2 tau
    const dtau = repmat(r.p.theta * (r.p.mu .- r.taus'), ng, 1) # theta (mu-tau)
    const dtau2 = repmat(2.0r.p.theta * r.p.sig^2 / r.p.mu * r.taus',
                         ng, 1)                            # theta sig^2 / mu tau
    zeros(ng, ntau), dg2, dtau, dtau2, zeros(ng, ntau),
    r.gs[2] - r.gs[1], r.taus[2] - r.taus[1]
end
coarser(r::GTauProcSpace, xdiv::Real, ydiv::Real) = 
    GTauProcSpace(r.p, int(div(length(r.gs),xdiv)), int(div(length(r.taus),ydiv)),
                  maximum(r.taus))


# ----------------------------------------------------------------------------
# PDE solvers to compute the expected value
# ----------------------------------------------------------------------------

# all solvers operate over a 2D space, equally spaced in each of the two
# dimensions (but spacing does not need to be equal across dimensions).
# It assumes the boundary conditions
# - constant boundaries in x
# - constant upper boundary in y
# - lose lower boundary in y, assuming <dx> = <dx2> = <dy2> = 0 and <dy> > 0

# PDE solver using single matrix inversion over collapsed 2D space
immutable PDE2DSolver <: AbstractPDE2DSolver
    ukj::Matrix{Float64}
    ukpj::Matrix{Float64}
    uknj::Matrix{Float64}
    ukjp::Matrix{Float64}
    ukjn::Matrix{Float64}
    ukdjd::Matrix{Float64}
    L::SparseMatrixCSC{Float64,Int}

    function PDE2DSolver{T <: Real}(dx::Matrix{T}, dx2::Matrix{T},
        dy::Matrix{T}, dy2::Matrix{T}, dxy::Matrix{T},
        Dx::Real, Dy::Real, dt::Real)
        Dx, Dy, dt = float(Dx), float(Dy), float(dt)
        Dx > 0.0 || error("Dx needs to be positive")
        Dy > 0.0 || error("Dy needs to be positive")
        dt > 0.0 || error("dt needs to be positive")
        const K, J = size(dx, 1), size(dx, 2)
        K == size(dy, 1) && J == size(dy, 2) || error("dx and dy need to be of same size")
        K == size(dx2, 1) && J == size(dx2, 2) || error("dx and dx2 need to be of same size")
        K == size(dy2, 1) && J == size(dy2, 2) || error("dx and dy2 need to be of same size")
        K == size(dxy, 1) && J == size(dxy, 2) || error("dx and dxy need to be of same size")
        const dtDx, dtDy, dtDx2, dtDy2 = dt/Dx, dt/Dy, dt/(Dx*Dx), dt/(Dy*Dy)
        const ukj = 0.5(dtDx2*float(dx2) .+ dtDy2*float(dy2))
        const ukpj = 0.25(dtDx*float(dx) .+ dtDx2*float(dx2))
        const uknj = 0.25(dtDx2*float(dx2) .- dtDx*float(dx))
        const ukjp = 0.25(dtDy*float(dy) .+ dtDy2*float(dy2))
        const ukjn = 0.25(dtDy2*float(dy2) .- dtDy*float(dy))
        const ukdjd = dt/(8.0Dx*Dy)*float(dxy)
        new(ukj, ukpj, uknj, ukjp, ukjn, ukdjd,
            _get2DSolverL(ukj, ukpj, uknj, ukjp, ukjn, ukdjd))
    end
end
function PDE2DSolver(r::AbstractProcSpace, dt::Real)
    dt > zero(dt) || error("dt needs to be positive")
    const dx, dx2, dy, dy2, dxy, Dx, Dy = moments(r, dt)
    PDE2DSolver(dx, dx2, dy, dy2, dxy, Dx, Dy, dt)
end
size(p::PDE2DSolver) = size(p.ukj)
expectedu{T <: Real}(p::PDE2DSolver, u::Matrix{T}) = 
    reshape(p.L \ _getb(p, float(u)), size(u, 1), size(u, 2))

# returns L matrix for PDE2DSolver
#
# The given matrices are of size K x J, containing various pre-computed
# constants.
#
# The returned sparse L is of size KJ x KJ
function _get2DSolverL(ukj, ukpj, uknj, ukjp, ukjn, ukdjd)
    const K, J = size(ukj)
    const maxels = 3K - 2 + 9(J-2)*(K-2) + 2(J-2)
    row, col, v = Array(Int, maxels), Array(Int, maxels), Array(Float64, maxels)
    # fill in-between block elements
    addnonzero(ri, ci, vi, i) = v == 0.0 ? i : begin   # only add non-zero elements
                                                   row[i], col[i], v[i] = ri, ci, vi
                                                   i + 1
                                               end
    # diagonals of first two block in top block row determine j=1 boundary
    row[1], col[1], v[1] = 1, 1, 1.0
    for k = 2:(K-1)
        const ki = 2k - 2
        row[ki], col[ki], v[ki] = k, k, 1.0 + 2.0ukjp[k,1]
        row[ki+1], col[ki+1], v[ki+1] = k, k+K, -2.0ukjp[k,1]
    end
    i = 2(K-1) + 1
    row[i-1], col[i-1], v[i-1] = K, K, 1.0
    # all j > 1 blocks
    for j = 2:(J-1)
        # first row has 1 along diagonal
        i = addnonzero((j-1)K + 1, (j-1)K + 1, 1.0, i)
        # other rows
        for k = 2:(K-1)
            const ri = (j-1)K + k  # row of L
            const ukdjdkj = ukdjd[k,j]
            # lower tri-diagonal block tri-diagonal elements
            i = addnonzero(ri, ri-K-1, -ukdjdkj, i)
            i = addnonzero(ri, ri-K, -ukjn[k,j], i)
            i = addnonzero(ri, ri-K+1, ukdjdkj, i)
            # diagonal block tri-diagonal elements
            i = addnonzero(ri, ri-1, -uknj[k,j], i)
            i = addnonzero(ri, ri, 1.0+ukj[k,j], i)
            i = addnonzero(ri, ri+1, -ukpj[k,j], i)
            # upper tri-diagonal block tri-diagonal elements
            i = addnonzero(ri, ri+K-1, -ukdjdkj, i)
            i = addnonzero(ri, ri+K, -ukjp[k,j], i)
            i = addnonzero(ri, ri+K+1, ukdjdkj, i)
        end
        # last row has 1 along diagonal
        i = addnonzero(j*K, j*K, 1.0, i)
    end
    # last block diagonal is identity
    const lastbase = (J-1)K
    for k = 1:K
        row[i+k-1], col[i+k-1], v[i+k-1] = lastbase + k, lastbase + k, 1.0
    end
    i += K - 1
    # return sparse L
    sparse(row[1:i], col[1:i], v[1:i], K*J, K*J)
end

# returns b vector for PDE2DSolver
#
# The given u is of size K x J, with x along rows, and y along columns.
#
# The returned b has KJ elements.
function _getb(p::PDE2DSolver, u::Matrix{Float64})
    const K, J = size(p)
    @assert K == size(u, 1) && J == size(u, 2)
    # fill b as b[k,j] and then reshape
    b = Array(Float64, K, J)
    # b[:,1] = u[:,1]
    # first block for j = 1 is different due to boundary conditions
    b[1,1] = u[1,1]
    for k = 2:(K-1)
        b[k,1] = (1.0 - 2.0p.ukjp[k,1])*u[k,1] + 2.0p.ukjp[k,1]*u[k,2]
    end
    b[K,1] = u[K,1]
    # for all j > 1
    for j = 2:(J-1)
        b[1,j] = u[1,j]
        for k = 2:(K-1)
            b[k,j] = (1.0-p.ukj[k,j])*u[k,j] +
                p.ukpj[k,j]*u[k+1,j] + p.uknj[k,j]*u[k-1,j] +
                p.ukjp[k,j]*u[k,j+1] + p.ukjn[k,j]*u[k,j-1] +
                p.ukdjd[k,j]*(u[k+1,j+1]+u[k-1,j-1]-u[k+1,j-1]-u[k-1,j+1])
        end
        b[K,j] = u[K,j]
    end
    b[:,J] = u[:,J]
    reshape(b, K*J)
end

# PDE solver using single matrix inversion over collapse 2D space
# This solver uses the ADI method and does not support <dx dy> != 0
immutable PDE2DADISolver <: AbstractPDE2DSolver
    ukjx::Matrix{Float64}
    ukjy::Matrix{Float64}
    ukpj::Matrix{Float64}
    uknj::Matrix{Float64}
    ukjp::Matrix{Float64}
    ukjn::Matrix{Float64}
    L1::Vector{Tridiagonal{Float64}}
    L2::Vector{Tridiagonal{Float64}}

    function PDE2DADISolver{T <: Real}(dx::Matrix{T}, dx2::Matrix{T},
        dy::Matrix{T}, dy2::Matrix{T}, Dx::Real, Dy::Real, dt::Real)
        Dx, Dy, dt = float(Dx), float(Dy), float(dt)
        Dx > 0.0 || error("Dx needs to be positive")
        Dy > 0.0 || error("Dy needs to be positive")
        dt > 0.0 || error("dt needs to be positive")
        const K, J = size(dx, 1), size(dx, 2)
        K == size(dy, 1) && J == size(dy, 2) || error("dx and dy need to be of same size")
        K == size(dx2, 1) && J == size(dx2, 2) || error("dx and dx2 need to be of same size")
        K == size(dy2, 1) && J == size(dy2, 2) || error("dx and dy2 need to be of same size")
        const dtDx, dtDy, dtDx2, dtDy2 = dt/Dx, dt/Dy, dt/(Dx*Dx), dt/(Dy*Dy)
        const ukjx = 0.5dtDx2*float(dx2)
        const ukjy = 0.5dtDy2*float(dy2)
        const ukpj = 0.25(dtDx*float(dx) .+ dtDx2*float(dx2))
        const uknj = 0.25(dtDx2*float(dx2) .- dtDx*float(dx))
        const ukjp = 0.25(dtDy*float(dy) .+ dtDy2*float(dy2))
        const ukjn = 0.25(dtDy2*float(dy2) .- dtDy*float(dy))
        const L1, L2 = _get2DADISolverL(ukjx, ukjy, ukpj, uknj, ukjp, ukjn)
        new(ukjx, ukjy, ukpj, uknj, ukjp, ukjn, L1, L2)
    end
end
size(p::PDE2DADISolver) = size(p.ukj)
function PDE2DADISolver(r::AbstractProcSpace, dt::Real)
    dt > zero(dt) || error("dt needs to be positive")
    const dx, dx2, dy, dy2, dxy, Dx, Dy = moments(r, dt)
    @assert all([isapprox(dxyi, 0.0) for dxyi in vec(dxy)])
    PDE2DADISolver(dx, dx2, dy, dy2, Dx, Dy, dt)
end


function expectedu{T <: Real}(p::PDE2DADISolver, u::Matrix{T})
    const K, J = size(u, 1), size(u, 2)
    @assert K == size(p.ukjx, 1) && J == size(p.ukjx, 2)
    # step 1 from n+1 to n+1/2
    u1 = Array(Float64, K, J)
    # boundary condition for j = 1
    u1[1,1] = u[1,1]
    for k = 2:(K-1)
        u1[k,1] = (1.0-2.0p.ukjp[k,1])*u[k,1] + 2.0p.ukjp[k,1]*u[k,2]
    end
    u1[K,1] = u[K,1]
    # 2 <= j <= J - 1
    bj = Array(Float64, K)
    for j = 2:(J-1)
        bj[1] = u[1,j]
        for k = 2:(K-1)
            @inbounds begin
                bj[k] = (1.0-p.ukjy[k,j])*u[k,j] +
                        p.ukjp[k,j]*u[k,j+1] + p.ukjn[k,j]*u[k,j-1]
            end
        end
        bj[K] = u[K,j]
        u1[:,j] = p.L1[j-1] \ bj
    end
    # boundary condition for j = J
    u1[:,J] = u[:,J]

    # step 2 from n+1/2 to n
    u2 = Array(Float64, K, J)
    # boundary condition for k = 1
    u2[1,:] = u1[1,:]
    # 2 <= k <= K - 1
    bk = Array(Float64, J)
    for k = 2:(K-1)
        bk[1] = u1[k,1]
        for j = 2:(J-1)
            @inbounds begin
                bk[j] = (1.0-p.ukjx[k,j])*u1[k,j] +
                        p.ukpj[k,j]*u1[k+1,j] + p.uknj[k,j]*u1[k-1,j]
            end
        end
        bk[J] = u1[k,J]
        u2[k,:] = p.L2[k-1] \ bk
    end
    # boundary condition for k = K
    u2[K,:] = u1[K,:]

    u2
end


function _get2DADISolverL(ukjx, ukjy, ukpj, uknj, ukjp, ukjn)
    const K, J = size(ukjx, 1), size(ukjx, 2)
    L1 = Array(Tridiagonal{Float64}, J-2)
    # L1 matrices for 2 <= j <= J-1
    for j = 2:(J-1)
        L1[j-1] = Tridiagonal([-uknj[2:(K-1),j], 0.0],
                              [1.0, 1.0.+ukjx[2:(K-1),j], 1.0],
                              [0.0, -ukpj[2:(K-1),j]])
    end
    L2 = Array(Tridiagonal{Float64}, K-2)
    # L2 matrices for 2 <= k <= K-1
    for k = 2:(K-1)
        L2[k-1] = Tridiagonal([-vec(ukjn[k,2:(J-1)]), 0.0],
                              [1.0+2.0ukjp[k,1], 1.0.+vec(ukjy[k,2:(J-1)]), 1.0],
                              [-2.0ukjp[k,1], -vec(ukjp[k,2:(J-1)])])
    end
    L1, L2
end


end # module
