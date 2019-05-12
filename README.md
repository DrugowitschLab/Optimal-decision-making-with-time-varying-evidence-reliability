# Optimal decision-making with time-varying evidence reliability

This code was used to generate the figures in Drugowitsch, Moreno-Bote & Pouget (2014) [1]. It is written in an old version of Julia (~0.3), and most likely won't run on a new version. It relies on equally old versions of the following libraries:
- `Distributions.jl`: probability distributions
- `Winston.jl`: plotting library
- `Color.jl`: color management (used for plotting)
- `HDF5.jl`: writing/reading files in HDF5 format
- `JLD.jl`: writing/reading files in Julia Language Data format
- `Roots.jl`: methods for root finding
- `NLopt.jl`: non-linear optimization library

## Files

- `vi2.jl`: defines the `VI2` module where the value iteration is happening
- `cirproc.jl`: defines the `CIRProc` module for simulations of the CIR process
- `rr_optim:jl` (included in `vi2.jl`): heuristic bound optimizations
- `cir_examples.jl`: simulates CIR process example trajectories
- `cir_examples_plot.jl`: plots CIR process example trojectories
- `dp2_examples.jl`: computes bound examples (Fig. 3)
- `dp2_examples_plot.jl`: plots bound examples
- `specialfuns.jl`: implements some special functions
- `utils.jl`: utility functions for file reading/writing & histograms

## Core code sections in `VI2` module

[l32](vi2.jl#L32): defines `FDTask` object, for fixed duration task  
[l47](vi2.jl#L47): defines `RTTask` object, for reaction time task

[l91](vi2.jl#L91): value iteration for `FDTask` (Fig. 1a)  
[l111](vi2.jl#L111): value iteration for `RTTask`, performing root finding on reward rate starting on a coarse grid, and then move to a fine grid (Fig. 1b) both call vinorm as core function to perform value iteration

[l199](vi2.jl#L199): `vinorm` creates the solver object and then performs value iteratino by calling `vinormiter`

[l224](vi2.jl#L224): `vinormiter` performs value iteration  
[l240](vi2.jl#L240): main loop (Fig. 1a) until convergence (checked in [l265](vi2.jl#L265))  
[l242](vi2.jl#L242): calls PDE solver to find expected value  
[l278](vi2.jl#L278): intersection of value functions (max in Eq. (5)) by interpolation

[l313](vi2.jl#L313): uses linear interpolation between grid points to find boundary location

[l352](vi2.jl#L352): computes `<V(1/2,tau)>` across tau's, as target for root finding (Fig. 1b)

[l370](vi2.jl#L370): discretization of spaced over which value function is defined  
[l375](vi2.jl#L375): space over `(X, tau)`  
[l420](vi2.jl#L420): space over `(g, tau)`

[l460](vi2.jl#L460): PDE solvers  
[l471](vi2.jl#L471): using single matrix inversion - can't recall if correct and just slower  
[l604](vi2.jl#L604): PDE solver described in SI of [1]  
[l631](vi2.jl#L631): pre-computes `L` matrices (Eqs. (25) and (29) in SI) as tridiagonal matrices  
[l644](vi2.jl#L644): compute expected `u`  
[l655](vi2.jl#L655): computes Eq. (28) in SI by assembling `u`, `b`, and then solving system  
[l671](vi2.jl#L671): computes Eq. (24) in SI by assembling `u`, `b`, abd then solving system  
[l695](vi2.jl#L695): computes L matrices by specifying (tri-)diagonal entries

## Citation

[1] Jan Drugowitsch, Ruben Moreno-Bote, and Alexandre Pouget.  
[Optimal decision-making with time-varying evidence reliability](https://papers.nips.cc/paper/5540-optimal-decision-making-with-time-varying-evidence-reliability).  
Advances in Neural Information Processing Systems 27 (NIPS 2014).  

