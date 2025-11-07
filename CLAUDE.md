# GANDALF: JAX-based KRMHD Spectral Solver

## Project Context
Writing a Journal of Plasma Physics paper on GANDALF, a modern reimplementation of Kinetic Reduced MHD turbulence solver using JAX for accessibility on commodity hardware (especially Apple Silicon).

## Key Collaborators
- Anjor Kanekar (lead developer, former collaborator with Schekochihin/Meyrand/Dorland)
- Paper builds on plasma echo work (Meyrand et al. 2019 PNAS)
- Editorial connection: Schekochihin & Loureiro on JPP board

## Paper Goals
1. Document GANDALF as accessible tool for turbulence research
2. Demonstrate spectral accuracy for KRMHD cascades
3. Verify against standard benchmarks
4. Enable broader participation in plasma turbulence research

## Technical Foundation

### Physics
- **KRMHD**: Kinetic Reduced MHD - captures Alfvén waves + slow modes
- **Regime**: Strong guide field, k∥ ≪ k⊥
- **Key physics**: Nonlinear cascades, phase mixing, helicity conservation

### Numerics
- Fourier spectral in perpendicular plane (x,y)
- Configurable parallel direction (z): spectral or finite difference
- Time stepping: RK4
- Dealiasing: 2/3 rule

### Implementation
- Pure JAX (no CUDA dependency)
- JIT compilation for performance
- Runs on CPU/GPU/TPU transparently
- Optimized for Apple Silicon Metal backend

## Benchmarks to Include
1. **Linear**: Alfvén wave dispersion
2. **Nonlinear**: Orszag-Tang vortex
3. **Turbulent**: k⊥^(-5/3) spectrum
4. **Conservation**: Energy/helicity in decay

## Paper Structure
1. Introduction - Physics motivation
2. Mathematical Formulation - KRMHD equations
3. Numerical Methods - Spectral approach
4. Implementation - JAX architecture
5. Verification - Benchmark results
6. Conclusions - Impact on field

## Writing Workflow
- Each section = separate PR
- Automated Claude review via GitHub Actions
- Master branch = latest complete draft
- Issues track remaining tasks

## Key Differentiators
- **NOT** another HPC code requiring clusters
- **NOT** competing with GS2/Gkeyll on features
- **IS** enabling solo researchers and small groups
- **IS** maintaining research-grade accuracy

## Important References
- Schekochihin et al. (2009) - KRMHD formulation
- Numata et al. (2010) - AstroGK code
- Meyrand et al. (2019) - Plasma echo physics
- Original GANDALF: github.com/anjor/gandalf-original

## Sub-Agents for Paper Writing

### Available Agents

#### `latex-equations`
- **Role**: LaTeX equation formatting and mathematical derivations
- **Use for**: KRMHD equations, dispersion relations, conservation laws
- **Maintains**: Consistency with notation.tex

#### `literature-curator`
- **Role**: Citation management and bibliography
- **Use for**: Finding references, formatting BibTeX, ensuring citation completeness
- **Key sources**: Schekochihin, Kunz, Loureiro, Dorland, Barnes, Howes

#### `benchmark-analyst`
- **Role**: Benchmark data analysis and figure generation
- **Use for**: Convergence plots, spectral analysis, comparison with theory
- **Output**: Publication-quality figures following JPP standards

#### `physics-narrator`
- **Role**: Physics exposition for expert audience
- **Use for**: Introduction, physical interpretations, connecting numerics to physics
- **Style**: Graduate-level plasma physics, no basic explanations needed

#### `code-documentor`
- **Role**: Implementation and algorithm documentation
- **Use for**: JAX optimizations, parallelization strategy, pseudocode
- **Focus**: Physics-relevant implementation choices

### Usage Patterns

**Single agent for focused task:**
```
@latex-equations: Format the gyrokinetic ordering equations
```

**Multiple agents for complete section:**
```
@physics-narrator: Write intro paragraph on cascade physics
@latex-equations: Add cascade equations
@literature-curator: Add Schekochihin 2009 and recent Parker Solar Probe papers
```

**Agent chain for verification:**
```
@benchmark-analyst: Generate Orszag-Tang vortex results
@physics-narrator: Explain physical significance
@latex-equations: Add convergence rate formula
```

### Agent Responsibilities Matrix

| Section | Primary Agent | Supporting Agents |
|---------|--------------|-------------------|
| Introduction | physics-narrator | literature-curator |
| Formulation | latex-equations | physics-narrator |
| Numerics | code-documentor | latex-equations |
| Verification | benchmark-analyst | physics-narrator, latex-equations |
| Implementation | code-documentor | - |
| Conclusions | physics-narrator | - |

### Quality Checks by Agents

- **latex-equations**: Validates all math symbols against notation.tex
- **literature-curator**: Ensures no unsupported claims
- **benchmark-analyst**: Verifies error convergence rates
- **physics-narrator**: Checks physics correctness and clarity
- **code-documentor**: Confirms code descriptions match actual implementation

## Writing Workflow with Sub-Agents

1. **Section Planning**: Identify which agents needed
2. **Initial Draft**: Primary agent writes section
3. **Enhancement**: Supporting agents add equations/citations
4. **Review**: All relevant agents review for consistency
5. **PR Creation**: Include agent contributions in commit messages

## Important Notes

- Agents assume JPP audience (expert plasma physicists)
- All agents reference GANDALF code at github.com/anjor/gandalf
- Benchmark data should be in `data/` directory
- Figures output to `paper/figures/` with descriptive names
- Each agent maintains its own context but shares notation.tex
