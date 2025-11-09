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
- Time stepping: GANDALF integrating factor method (RK2 midpoint with exact linear wave propagation)
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

## Scientific Writing Guidelines

### Voice & Style
- **Authoritative but accessible**: Write as an expert addressing experts
- **Active voice preferred**: "We solve the KRMHD equations" not "The KRMHD equations are solved"
- **Present tense for the code**: "GANDALF employs spectral methods"
- **Past tense for completed work**: "We benchmarked against Orszag-Tang"
- **Avoid hedge words**: Not "might possibly show" → "demonstrates"
- **Quantitative over qualitative**: "reduces error by factor of 10³" not "significantly reduces error"

### Structure Rules
- Start sections with the main point, then elaborate
- One idea per paragraph, topic sentence first
- Equations should flow with text, not interrupt it
- Every figure must advance the argument

### Technical Writing
- Define acronyms on first use: "Kinetic Reduced MHD (KRMHD)"
- Consistent notation throughout (use notation.tex file)
- Number ALL equations that are referenced
- Describe what equations mean physically, not just mathematically

### LaTeX Specifications

**Document Setup:**
```latex
\documentclass[12pt]{article}
\usepackage{jpp} % JPP style file if available
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx,subcaption}
\usepackage{hyperref}
\usepackage{cleveref} % use \cref{} for references
```

**Equation Formatting:**
- Align multi-line equations at = signs
- Use \begin{subequations} for related equation groups
- Define custom commands for repeated terms:
```latex
\newcommand{\kperp}{k_\perp}
\newcommand{\kpar}{k_\parallel}
```

**Figure Guidelines:**
- Vector formats (PDF/EPS) for line plots
- 300 DPI minimum for raster images
- Sans-serif fonts in figures (Helvetica/Arial)
- Label axes with units: "$k_\perp \rho_i$"

**Citation Style:**
- Use BibTeX with JPP bibliography style
- Cite as: "\citet{Schekochihin2009} demonstrated..." or "...was shown \citep{Schekochihin2009}"
- Group citations chronologically: \citep{Howes2006,Schekochihin2009,Kunz2018}

### JPP Journal Requirements
- Abstract: 200 words maximum
- Include 5-6 keywords
- Sections: Introduction, Formulation, Numerical Method, Results, Discussion, Conclusions
- Acknowledgments before references
- Data availability statement required

### Code Description Guidelines
When describing GANDALF:
1. Emphasize JAX benefits without making it a JAX advertisement
2. Compare with existing codes (Viriato, GS2, AstroGK) respectfully
3. Highlight accessibility without disparaging HPC approaches
4. Include code availability statement with GitHub URL

### Benchmark Presentation
For each benchmark:
1. State the physical test
2. Give expected analytical/theoretical result
3. Show GANDALF results
4. Quantify agreement (convergence plots, error scaling)

### Review Criteria for PRs
Before approving any section, verify:
- [ ] Physics correctness
- [ ] Mathematical rigor
- [ ] Consistent notation
- [ ] All claims supported by evidence
- [ ] Figures/equations properly referenced
- [ ] No undefined terms
- [ ] Smooth transitions between paragraphs

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
Use the latex-equations agent: Format the gyrokinetic ordering equations
```

**Multiple agents for complete section:**
```
Use the physics-narrator agent: Write intro paragraph on cascade physics
Use the latex-equations agent: Add cascade equations
Use the literature-curator agent: Add Schekochihin 2009 and recent Parker Solar Probe papers
```

**Agent chain for verification:**
```
Use the benchmark-analyst agent: Generate Orszag-Tang vortex results
Use the physics-narrator agent: Explain physical significance
Use the latex-equations agent: Add convergence rate formula
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
