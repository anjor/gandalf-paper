# GANDALF Paper Writing Instructions

## Project Overview
You are writing a computational physics paper about GANDALF, a JAX-based spectral solver for KRMHD turbulence. Target journal: Journal of Plasma Physics (JPP).

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

## LaTeX Specifications

### Document Setup
```latex
\documentclass[12pt]{article}
\usepackage{jpp} % JPP style file if available
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx,subcaption}
\usepackage{hyperref}
\usepackage{cleveref} % use \cref{} for references
```

### Equation Formatting
- Align multi-line equations at = signs
- Use \begin{subequations} for related equation groups
- Define custom commands for repeated terms:
```latex
  \newcommand{\kperp}{k_\perp}
  \newcommand{\kpar}{k_\parallel}
```

### Figure Guidelines
- Vector formats (PDF/EPS) for line plots
- 300 DPI minimum for raster images
- Sans-serif fonts in figures (Helvetica/Arial)
- Label axes with units: "$k_\perp \rho_i$"

### Citation Style
- Use BibTeX with JPP bibliography style
- Cite as: "\citet{Schekochihin2009} demonstrated..." or "...was shown \citep{Schekochihin2009}"
- Group citations chronologically: \citep{Howes2006,Schekochihin2009,Kunz2018}

## JPP Specific Requirements
- Abstract: 200 words maximum
- Include 5-6 keywords
- Sections: Introduction, Formulation, Numerical Method, Results, Discussion, Conclusions
- Acknowledgments before references
- Data availability statement required

## Code-Specific Instructions

When describing GANDALF:
1. Emphasize JAX benefits without making it a JAX advertisement
2. Compare with existing codes (Viriato, GS2, AstroGK) respectfully
3. Highlight accessibility without disparaging HPC approaches
4. Include code availability statement with GitHub URL

## Benchmark Presentation
For each benchmark:
1. State the physical test
2. Give expected analytical/theoretical result
3. Show GANDALF results
4. Quantify agreement (convergence plots, error scaling)

## Review Criteria for PRs
Before approving any section, verify:
- [ ] Physics correctness
- [ ] Mathematical rigor
- [ ] Consistent notation
- [ ] All claims supported by evidence
- [ ] Figures/equations properly referenced
- [ ] No undefined terms
- [ ] Smooth transitions between paragraphs
