# GANDALF Paper: An AI-Assisted Scientific Research Demonstration

> **Testing the Intelligence Explosion**: Can AI turn one physicist into a research team?

This repository contains a complete scientific paper and verification study demonstrating AI-assisted research at scale. Both the [GANDALF code](https://github.com/anjor/gandalf) and this paper were primarily written by AI, guided by domain expertise.

📖 **Read the full story**: [Testing the Intelligence Explosion: Can AI Turn One Physicist Into a Research Team?](https://anjor.xyz/writing/2025/11/05/testing-the-intelligence-explosion-can-ai-turn-one-physicist-into-a-research-team/)

📄 **Read the paper**: [GANDALF_paper.pdf](GANDALF_paper.pdf) (pre-compiled, ready to view)

---

## The Experiment

This project tests whether AI can produce **research-grade scientific work**, not just toy examples. The deliverables include:

- **A production code**: [GANDALF](https://github.com/anjor/gandalf) - 2,000+ lines of JAX implementing spectral KRMHD solver
- **This paper**: ~1,000 lines of LaTeX for Journal of Plasma Physics submission
- **Rigorous verification**: Three benchmark suites demonstrating spectral accuracy
- **Publication-quality figures**: 12 figures generated from numerical experiments
- **Automated quality control**: GitHub Actions with Claude-based review workflows

### What's AI-Generated?

**Nearly everything:**
- Complete JAX codebase (spectral methods, time integration, parallelization)
- All paper sections (introduction through conclusions)
- Mathematical formulation and LaTeX equations
- Benchmark analysis and figure generation
- Documentation and workflow automation

**Human contributions:**
- Scientific direction and physics expertise
- Verification criteria and benchmark selection
- Quality review and refinement
- Connection to existing research literature

---

## The Science

### GANDALF: GPU-Accelerated Numeric Dynamics of Alfvénic Turbulence

GANDALF is a modern reimplementation of Kinetic Reduced MHD (KRMHD) turbulence physics using JAX. It enables plasma turbulence research on **commodity hardware** (laptops, desktops, Apple Silicon) without requiring HPC clusters.

**Key Features:**
- Fourier spectral discretization in perpendicular plane
- Hermite spectral expansion in parallel velocity
- GANDALF integrating factor time stepping (exact linear propagation)
- Transparent CPU/GPU/TPU execution via JAX
- Research-grade accuracy with ~2-3× performance trade-off vs CUDA codes

**Physical Regime:**
- Strong guide field: k‖ ≪ k⊥
- Alfvénic turbulence and phase mixing
- Solar wind, magnetospheric, and fusion plasma applications

### Verification Benchmarks

This paper demonstrates three levels of validation:

1. **Linear**: Alfvén wave dispersion relations (machine precision: ~10⁻¹⁵ error)
2. **Nonlinear**: Orszag-Tang vortex evolution (10⁻⁶ energy conservation)
3. **Turbulent**: Forced cascade with k⊥⁻⁵/³ spectrum over 200 Alfvén times
4. **Velocity Space**: Phase mixing cascade with m⁻¹/² scaling

Each benchmark includes convergence studies and comparison with analytical/theoretical predictions.

---

## Repository Structure

```
gandalf-paper/
├── paper/
│   ├── main.tex                    # Master LaTeX document
│   ├── sections/
│   │   ├── introduction.tex        # Physics motivation
│   │   ├── formulation.tex         # KRMHD equations
│   │   ├── numerics.tex            # Spectral methods
│   │   ├── implementation.tex      # JAX architecture
│   │   ├── verification.tex        # Benchmark results
│   │   ├── discussion.tex          # Interpretation and ecosystem positioning
│   │   └── conclusions.tex         # Impact and future work
│   ├── figures/                    # Publication-quality figures (12 PDFs/PNGs)
│   ├── references.bib              # Bibliography
│   ├── notation.tex                # Mathematical notation definitions
│   └── main.pdf                    # Compiled paper
├── data/
│   └── benchmarks/                 # Raw benchmark data
│       ├── alfven_wave/
│       ├── orszag_tang/
│       └── turbulent_cascade/
├── scripts/                        # Python analysis and plotting scripts
├── .github/workflows/              # Automated quality control
│   ├── claude-code-review.yml      # Claude-based code review
│   ├── notation-check.yml          # Notation consistency checks
│   └── paper-review.yml            # Paper quality checks
└── CLAUDE.md                       # Detailed project guidelines
```

---

## Building the Paper

**Quick start**: A pre-compiled PDF is available at the root: [GANDALF_paper.pdf](GANDALF_paper.pdf)

To rebuild from source:

### Prerequisites

```bash
# LaTeX distribution (TeX Live, MacTeX, or MiKTeX)
# Required packages: amsmath, amssymb, graphicx, hyperref, cleveref

# Python environment for regenerating figures
pip install jax numpy scipy matplotlib h5py
```

### Compile the Paper

```bash
cd paper/
make                # Compiles main.tex → main.pdf
make clean          # Removes auxiliary files
```

Or manually:
```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The compiled PDF will be `paper/main.pdf`.

---

## Reproducibility

### Scientific Reproducibility

All benchmark results can be regenerated from the [GANDALF code](https://github.com/anjor/gandalf):

```bash
# Clone the GANDALF repository
git clone https://github.com/anjor/gandalf.git
cd gandalf

# Run benchmarks (see GANDALF README for configuration)
python examples/alfven_wave_benchmark.py
python examples/orszag_tang_benchmark.py
python examples/turbulent_cascade_benchmark.py
```

Raw data is archived in `data/benchmarks/` for figure regeneration without re-running simulations.

### Workflow Reproducibility

This repository demonstrates an **AI-assisted research workflow**:

1. **Development**: AI generates code/text with human guidance
2. **Verification**: Automated tests and benchmark validation
3. **Review**: GitHub Actions with Claude-based quality checks
4. **Iteration**: PR-based workflow with AI code review
5. **Refinement**: Human physicist reviews for correctness and clarity

The `.github/workflows/` directory contains automation examples. See `CLAUDE.md` for detailed guidelines used to direct AI contributions.

---

## Paper Status

- **Submission Target**: Journal of Plasma Physics
- **Current Status**: Pre-submission (final review in progress)
- **Preprint**: Coming soon

### Citation

```bibtex
@article{kanekar2025gandalf,
  title={GANDALF: GPU-Accelerated Numeric Dynamics of Alfv\'enic Turbulence with JAX},
  author={Kanekar, Anjor},
  journal={Journal of Plasma Physics},
  year={2025},
  note={In preparation}
}
```

---

## Related Resources

- **Main Code Repository**: [github.com/anjor/gandalf](https://github.com/anjor/gandalf)
- **Blog Post**: [Testing the Intelligence Explosion](https://anjor.xyz/writing/2025/11/05/testing-the-intelligence-explosion-can-ai-turn-one-physicist-into-a-research-team/)
- **JAX Documentation**: [jax.readthedocs.io](https://jax.readthedocs.io/)
- **Journal of Plasma Physics**: [cambridge.org/jpp](https://www.cambridge.org/core/journals/journal-of-plasma-physics)

### Key Scientific References

- Schekochihin et al. (2009) - KRMHD formulation ([ApJS 182:310](https://doi.org/10.1088/0067-0049/182/1/310))
- Numata et al. (2010) - AstroGK gyrokinetic code ([JCP 229:9347](https://doi.org/10.1016/j.jcp.2010.09.006))
- Meyrand et al. (2019) - Plasma echo physics ([PNAS 116:1185](https://doi.org/10.1073/pnas.1813913116))

---

## Contributing

This paper uses a **PR-based workflow** with automated quality checks:

1. Create a feature branch: `git checkout -b feature-name`
2. Make changes to `paper/sections/*.tex`
3. Submit PR to `main` branch
4. Claude-based review runs automatically
5. Address feedback and merge

See `CLAUDE.md` for detailed writing guidelines (notation, style, figure formatting).

---

## What This Demonstrates

### AI Capabilities

✅ **Can AI produce research-grade work?** Yes, with proper guidance.

- Mathematical rigor: Correct KRMHD formulation with consistent notation
- Numerical correctness: Spectral methods with verified convergence rates
- Scientific writing: JPP-standard exposition for expert audience
- Software engineering: Production-quality JAX implementation

### AI Limitations

⚠️ **What still requires human expertise:**

- Scientific direction and research questions
- Physical intuition for interpreting results
- Connection to broader research context
- Quality judgment and refinement
- Ethical oversight and verification

### Insights

1. **AI as force multiplier**: One physicist can produce the work of a small research team
2. **Iterative refinement essential**: Initial AI output requires multiple rounds of review
3. **Domain expertise critical**: Guiding AI requires deep understanding of the field
4. **Automation accelerates iteration**: GitHub Actions enable rapid feedback cycles
5. **Transparency matters**: Clear documentation of AI contributions builds trust

---

## License

This paper and associated materials are available under MIT License. See [LICENSE](LICENSE) for details.

The GANDALF code is separately licensed (see [main repository](https://github.com/anjor/gandalf)).

---

## Acknowledgments

This work was made possible by:
- Claude (Anthropic) for AI assistance in code and paper generation
- JAX team (Google) for the numerical computing framework
- Plasma physics community for theoretical foundations

Built on research by Schekochihin, Kunz, Loureiro, Dorland, Barnes, Howes, and many others in the plasma turbulence community.

---

## Questions?

For questions about:
- **The AI experiment**: See [blog post](https://anjor.xyz/writing/2025/11/05/testing-the-intelligence-explosion-can-ai-turn-one-physicist-into-a-research-team/)
- **The physics**: Read `paper/main.pdf` or open an issue
- **The code**: Visit [github.com/anjor/gandalf](https://github.com/anjor/gandalf)
- **Replicating this approach**: See `CLAUDE.md` and `.github/workflows/`
