# Makefile for GANDALF paper
# Main LaTeX file (without .tex extension)
MAIN = main

# Output PDF name
PDF = $(MAIN).pdf

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# LaTeX compiler flags
LATEXFLAGS = -interaction=nonstopmode -halt-on-error

# Auxiliary files to clean
AUX_FILES = *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk \
            *.synctex.gz *.nav *.snm *.vrb *.bcf *.run.xml *.nlo *.nls \
            *.ilg *.ind *.idx *.glo *.gls *.glg *.acn *.acr *.alg

.PHONY: all pdf clean cleanall help

# Default target
all: pdf

# Build the PDF with full bibliography compilation
pdf: $(PDF)

$(PDF): $(MAIN).tex sections/*.tex notation.tex references.bib
	@echo "=== First LaTeX pass ==="
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	@echo ""
	@echo "=== Running BibTeX ==="
	-$(BIBTEX) $(MAIN)
	@echo ""
	@echo "=== Second LaTeX pass ==="
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	@echo ""
	@echo "=== Third LaTeX pass ==="
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	@echo ""
	@echo "=== PDF generated successfully: $(PDF) ==="

# Quick compile (single pass, useful during editing)
quick:
	@echo "=== Quick compile (single pass) ==="
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex

# Clean auxiliary files but keep PDF
clean:
	@echo "Cleaning auxiliary files..."
	rm -f $(AUX_FILES)
	@echo "Clean complete. PDF preserved."

# Clean everything including PDF
cleanall: clean
	@echo "Removing PDF..."
	rm -f $(PDF)
	@echo "All files cleaned."

# Alternative name for cleanall
distclean: cleanall

# Watch for changes and recompile (requires latexmk)
watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(MAIN).tex

# Display help
help:
	@echo "GANDALF Paper Makefile"
	@echo "====================="
	@echo ""
	@echo "Available targets:"
	@echo "  make          - Build the PDF (default)"
	@echo "  make pdf      - Build the PDF with full compilation"
	@echo "  make quick    - Quick single-pass compilation"
	@echo "  make clean    - Remove auxiliary files, keep PDF"
	@echo "  make cleanall - Remove all generated files including PDF"
	@echo "  make distclean- Same as cleanall"
	@echo "  make watch    - Auto-recompile on changes (requires latexmk)"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Main file: $(MAIN).tex"
	@echo "Output PDF: $(PDF)"
