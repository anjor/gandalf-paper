# Resurrecting Gandalf

<Prompt>
I want a simple code that can simulate kinetic reduced mhd equations

This is the code I wrote for my PhD (15 years ago), and haven't looked at it in the last 10-12 years. https://github.com/anjor/gandalf

I will be running it on commodity hardware: M1/M2 macbook pros, using infra providers like modal or baseten, or if none of those work then aws ec2 instances.

Questions:
1. What will it take to resurrect gandalf? Is there enough structure there to be of value or is it all throwaway?
2. What would it take to write such a code from scratch?
3. How to go about making a decision of fixing gandalf, vs writing gandalf 2.0 from scratch?
4. The current gandalf code uses nvidia gpus with cuda. How to run this on modern macbooks -- can i use mlx?
</Prompt>

<Response>
Looking at your GANDALF code - it's a solid spectral KRMHD implementation with good structure (Fourier/Hermite basis, proper normalization, energy diagnostics). Here's my assessment:

## 1. Resurrecting GANDALF
**Salvageable components:**
- Core physics implementation (KRMHD equations, spectral methods)
- Hermite polynomial infrastructure
- Diagnostic framework
- Test problems/validation cases

**Major blockers:**
- CUDA 2.x/3.x era code - significant API changes
- cuFFT interface has evolved substantially
- Fortran 90 + CUDA C hybrid is painful to maintain
- No modern build system (cmake/meson)

## 2. From scratch effort
~2-3 months for a functional KRMHD code with modern stack:
- Julia/Python + JAX/Triton for GPU
- Or pure C++20/23 with Kokkos for portability
- Spectral methods are straightforward to implement

## 3. Decision framework
**Resurrect if:** You need results in <1 month and can access NVIDIA hardware
**Rewrite if:** You want maintainable code for next 5+ years of research

My recommendation: **Rewrite using JAX or Julia**
- JAX runs on M1/M2 via Metal Performance Shaders
- Automatic differentiation for free
- Clean separation of physics from numerics
- 10x less code than Fortran+CUDA

## 4. Running on M1/M2
- CUDA won't work directly
- MLX is too limited (lacks complex FFTs)
- **JAX** is your best bet - full Metal backend
- Alternatively: Julia with Metal.jl

Want me to sketch out a minimal JAX-based KRMHD implementation structure?
</Response>

<Prompt>
I plan to use claude code for the development. Write out an implementation plan for claude code
</Prompt>

<Response>
## KRMHD Implementation Plan for Claude Code

### Phase 1: Core Infrastructure (Week 1)
**Goal:** Minimal working spectral framework

```
Task 1.1: Project setup and validation
- Create project structure with pyproject.toml
- Set up JAX with Metal backend for M1/M2
- Implement basic 2D Fourier transforms
- Verify FFT performance on test arrays

Task 1.2: Spectral operators
- Implement dealiasing (2/3 rule)
- Create gradient operators in Fourier space
- Add Poisson solver (-k²φ = ρ)
- Write unit tests for each operator

Task 1.3: Hermite basis for parallel velocity
- Port Hermite polynomial generation from GANDALF
- Implement Hermite transform pair
- Validate against analytical test functions
```

### Phase 2: KRMHD Physics (Week 2)
**Goal:** Implement core equations

```
Task 2.1: Field evolution
- Implement parallel magnetic field evolution (∂A∥/∂t)
- Add perpendicular magnetic field (∂B⊥/∂t)
- Include electron pressure gradient effects

Task 2.2: Kinetic pressure evolution
- Implement pressure evolution equation
- Add Landau damping closure
- Include FLR corrections if needed

Task 2.3: Timestepping
- Implement RK4 timestepper
- Add adaptive timestep with CFL condition
- Create checkpoint/restart functionality
```

### Phase 3: Validation Suite (Week 3)
**Goal:** Physics validation against known results

```
Task 3.1: Linear physics
- Alfvén wave propagation
- Kinetic Alfvén wave dispersion
- Landau damping rates

Task 3.2: Nonlinear benchmarks
- 2D turbulence decay
- Energy conservation tests
- Compare cascade rates with published results

Task 3.3: Performance optimization
- Profile with JAX profiler
- Optimize memory usage
- Implement mixed precision where appropriate
```

### Phase 4: Production Features (Week 4)
**Goal:** Research-ready code

```
Task 4.1: Diagnostics
- Energy spectra (magnetic, kinetic, total)
- Flux measurements
- Structure functions
- Phase space visualizations

Task 4.2: I/O and postprocessing
- HDF5 output with compression
- Restart capabilities
- Real-time plotting during runs
- Parallel I/O for large runs

Task 4.3: Parameter studies
- Config file system (YAML/TOML)
- Batch run capabilities
- Integration with Modal/Baseten for cloud runs
```

### Claude Code Strategy

**Prompt Engineering:**
```python
# Start each session with:
"""
You are implementing a spectral KRMHD solver in JAX.
Key requirements:
- Must run on Apple M1/M2 (Metal backend)
- Use spectral methods (Fourier + Hermite)
- Follow numpy/scipy conventions where possible
- Optimize for clarity over premature optimization
Reference implementation: github.com/anjor/gandalf
"""
```

**Incremental Development Pattern:**
1. Write minimal working version first
2. Add physics complexity incrementally
3. Validate each component before moving on
4. Keep functions under 50 lines for Claude Code context

**Testing Strategy:**
- Test each physics component in isolation
- Use manufactured solutions where possible
- Compare against GANDALF results for validation

**Code Review Checkpoints:**
After each phase, have Claude Code:
- Review for numerical stability
- Check energy conservation
- Verify parallelization opportunities
- Suggest performance improvements

Would you like me to detail any specific phase or provide example prompts for Claude Code to get started?
</Response>

<Prompt>
any hooks or agents or slash commands you would recommend?
</Prompt>

<Response>
For your KRMHD solver project, here are the most useful Claude Code features:

## Essential Slash Commands

**`/debug`** - Critical for numerical code
- Helps identify numerical instabilities
- Traces through array shape mismatches (common in spectral codes)
- Catches JAX-specific issues like tracers escaping

**`/test`** - Run validation suite frequently
- Set up to run your linear physics tests
- Energy conservation checks
- Spectral accuracy verification


**`/lint`** - Maintain code quality
- JAX has specific patterns (no side effects)
- Catches common issues like in-place operations

## Recommended Agents

**1. Code Review Agent**
```yaml
# .claude/agents/physics_reviewer.yaml
name: Physics Reviewer
description: Reviews code for physics correctness and numerical stability
instructions: |
  - Check energy conservation implementation
  - Verify spectral operations preserve reality conditions
  - Ensure dealiasing is applied after all nonlinear terms
  - Validate CFL condition includes all wave speeds
```

**2. Performance Optimization Agent**
```yaml
# .claude/agents/jax_optimizer.yaml  
name: JAX Performance Optimizer
description: Optimizes JAX code for Metal backend
instructions: |
  - Ensure all hot loops use jax.vmap or jax.scan
  - Check JIT compilation boundaries
  - Identify unnecessary CPU-GPU transfers
  - Suggest batching strategies for FFTs
```

## Custom Hooks

**Pre-commit hook for physics invariants:**
```python
# .claude/hooks/check_physics.py
def pre_commit_hook():
    """Verify physics constraints before committing"""
    checks = [
        "assert_energy_conservation",
        "verify_reality_conditions", 
        "check_hermite_orthogonality"
    ]
    # Run validation tests
```

**Run hook for diagnostics:**
```python
# .claude/hooks/run_diagnostics.py
def post_run_hook():
    """Auto-generate diagnostic plots after each run"""
    # Energy spectrum plots
    # Time series of invariants
    # Phase space visualization
```

## Workflow Aliases

```bash
# .claude/config.yaml
aliases:
  validate: "/test validation/*.py"
  bench: "/run benchmarks/linear_physics.py"
  profile: "/run --profile krmhd/timestepping.py"
  conserve: "/debug --check-conservation"
```

These tools will catch most KRMHD-specific issues early and maintain code quality throughout development.
</Response>

<Prompt>
great, i am about to start up claude code for this. Shall we write a separate implementation plan that claude code can follow and do step by step?
</Prompt>

<Response>
Yes! Here's a step-by-step implementation plan optimized for Claude Code's iterative development:

## KRMHD Solver: Step-by-Step Implementation Plan

### Step 1: Project Bootstrap
```bash
# First command to Claude Code
Create a new Python project structure for a KRMHD spectral solver:
- pyproject.toml with JAX, h5py, matplotlib dependencies  
- krmhd/ package with __init__.py
- tests/ directory
- examples/ directory
- README.md with basic project description
Ensure JAX will work on Apple M1/M2 Metal backend.
```

### Step 2: Core Spectral Infrastructure
```bash
Create krmhd/spectral.py with:
1. A SpectralField2D class that wraps JAX arrays
2. FFT/IFFT operations using jax.numpy.fft
3. Derivative operators (∂x, ∂y) in Fourier space
4. Dealising function (2/3 rule)
Include a simple test that verifies ∂x(sin(x)) = cos(x)
```

### Step 3: Poisson Solver
```bash
In krmhd/spectral.py add:
1. Poisson solver: solve ∇²φ = ω for φ given vorticity ω
2. Use k² in Fourier space (handle k=0 carefully)
3. Test with manufactured solution ω = sin(x)cos(y)
Verify the solver returns φ with correct boundary conditions.
```

### Step 4: Poisson Bracket
```bash
Create krmhd/physics.py with:
1. poisson_bracket(f, g) function computing {f,g} = ẑ·(∇f × ∇g)
2. Use spectral derivatives and proper dealiasing
3. Test with f=sin(x), g=cos(y) - should give sin(x)sin(y)
This is the core nonlinearity in KRMHD.
```

### Step 5: KRMHD State and Initialization
```bash
In krmhd/physics.py add:
1. KRMHDState dataclass with fields: phi, A_parallel, B_parallel, time
2. initialize_alfven_wave() function for linear wave test
3. initialize_random_spectrum() for turbulence with specified k^-α spectrum
4. energy() function computing E_magnetic + E_kinetic
```

### Step 6: Time Evolution (Alfvén Dynamics)
```bash
In krmhd/physics.py add the RHS for active fields:
1. dphi_dt computing RHS of stream function equation
2. dA_parallel_dt computing RHS of vector potential equation
3. Include viscosity/resistivity terms
4. Test: initialized Alfvén wave should propagate at correct speed
```

### Step 7: Passive Scalar Evolution
```bash
Extend krmhd/physics.py with slow mode evolution:
1. dB_parallel_dt for parallel magnetic field (passive scalar)
2. Ensure it's only advected, no back-reaction on phi or A_parallel
3. Test: B_parallel should be passively mixed by turbulent flow
```

### Step 8: Time Integrator
```bash
Create krmhd/timestepping.py with:
1. RK4 integrator that takes (state, dt, rhs_function)
2. CFL condition calculator based on max(v_Alfven, v_flow)
3. Test with Alfvén wave - verify 4th order convergence
```

### Step 9: Basic Diagnostics
```bash
Create krmhd/diagnostics.py with:
1. energy_spectrum(state) returning k, E(k)
2. energy_history tracking total, magnetic, kinetic energy
3. plot_state() for quick visualization of phi, A_parallel, B_parallel
Test by running decaying turbulence and plotting E(t)
```

### Step 10: Linear Physics Validation
```bash
Create tests/test_linear_physics.py with:
1. Alfvén wave dispersion test (ω vs k_parallel)
2. Check wave propagates at v_A
3. Verify energy conservation to machine precision
This validates our basic physics implementation.
```

### Step 11: Orszag-Tang Vortex
```bash
Create examples/orszag_tang.py:
1. Classic MHD test problem setup
2. Run to t=1.0, compare with known results
3. Check for correct shock formation
This tests nonlinear dynamics.
```

### Step 12: Decaying Turbulence Run
```bash
Create examples/decaying_turbulence.py:
1. Initialize with k^-1 spectrum
2. Run until cascade develops
3. Verify k^-5/3 inertial range
4. Check slow modes remain passive
This is our first physics research run.
```

### Step 13: HDF5 I/O
```bash
Create krmhd/io.py with:
1. save_checkpoint(state, filename) 
2. load_checkpoint(filename)
3. save_diagnostics(diagnostics_dict, filename)
Test with a turbulence run - checkpoint and restart.
```

### Step 14: Landau Damping Closure
```bash
Extend krmhd/physics.py with:
1. Simple Landau damping model for electron pressure
2. Add damping term to appropriate equations
3. Verify damping rate matches theory for linear waves
This completes the physics model.
```

### Step 15: Production Features
```bash
Create krmhd/config.py with:
1. YAML configuration file parser
2. Parameter validation
3. Run script that takes config file as input
Create example configs for standard cases.
```

---

**Execution notes for Claude Code:**
- Complete each step fully before moving to the next
- Run tests immediately after implementing each feature
- If a test fails, debug before proceeding
- Keep functions small and focused
- Use @jax.jit liberally but test without it first

Start with Step 1 and proceed sequentially. Each step builds on the previous ones, so order matters.
</Response>
