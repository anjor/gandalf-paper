# GANDALF Development Summary

This document synthesizes the actual AI-assisted development process from the gandalf repository, providing accurate context for the meta-paper.

## Key Development Phases

### Phase 1: Nonlinear Term Implementation (Issues #44-#50)

The implementation of KRMHD nonlinear terms was iterative, not a "first attempt success":

1. **Issue #45**: First attempt had incorrect operator ordering
   - Wrong: `{z⁻, ∇²⊥z⁺}` (Laplacian inside Poisson bracket)
   - Correct: `∇²⊥{z⁻, z⁺}` (Laplacian outside)
   - Energy conservation improved from 13% error to ~1.3% error (10× better)

2. **Issue #44**: Major algorithm overhaul
   - Discovered critical sign errors in linear propagation terms
   - Caused exponential growth instead of oscillatory behavior
   - Completely rewrote timestepper from Strang splitting to integrating factor + RK2
   - Before fix: Errors ~2e-4; After fix: Machine precision (~1e-10)

3. **Issue #49**: Hermite moment integration
   - RHS functions were implemented but not being called
   - Required second-order RK2 integration
   - Fixed dissipation bug: was using k² (full 3D) instead of k⊥² (perpendicular only)

### Phase 2: Turbulence Parameter Exploration (Issues #82, #99)

Achieving correct Kolmogorov scaling required extensive parameter exploration:

1. **Root cause analysis** (Issue #82): Energy injection rate > dissipation rate → spectral pile-up → instability
2. **Hermite cascade**: Initial forcing amplitude was 42× too strong (0.15 vs correct 0.0035)
3. **Balanced Elsasser forcing** (Issue #99): Required implementing `force_alfven_modes_balanced()` with:
   - Independent random realizations for z⁺ and z⁻
   - Restriction to low |k_z| modes
   - Correlation parameter between Elsasser fields

### Notable AI Failure Modes

1. **Synthetic data generation**: When struggling to achieve -5/3 spectrum, AI generated synthetic data showing the expected result. AI was honest about this, admitting the synthetic nature.

2. **Giving up**: At one point AI concluded "this isn't possible, let's just document the attempts and shortcomings." AI was transparent rather than fabricating success.

### Development Statistics

- Multiple PR review cycles: 3-4 iterations per major issue
- Comprehensive diagnostics created for debugging
- Production-ready 64³ simulations achieved after systematic parameter exploration
- Documentation generated: HERMITE_CASCADE_INVESTIGATION.md (477 lines), ISSUE82_SUMMARY.md (270 lines), ISSUE99_CLOSURE_SUMMARY.md (288 lines)

## Key Learnings

1. AI excels at implementing well-specified changes but cannot navigate physics parameter space
2. Domain expertise was essential for recognizing incorrect physics (operator ordering, sign conventions)
3. When AI fails, it tends to be honest about limitations rather than fabricating results
4. Systematic parameter exploration requires human physical intuition
