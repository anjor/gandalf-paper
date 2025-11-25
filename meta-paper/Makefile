# Makefile for meta-paper (AI-augmented physics paper)

MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

# Default target
all: $(MAIN).pdf

# Build PDF with bibliography
$(MAIN).pdf: $(MAIN).tex references.bib appendix.tex figures/autonomy-gradient.tex figures/workflow.tex
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

# Quick build (no bibliography update)
quick:
	$(LATEX) $(MAIN)

# Clean auxiliary files
clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out
	rm -f $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).fls $(MAIN).fdb_latexmk
	rm -f $(MAIN).synctex.gz

# Clean everything including PDF
distclean: clean
	rm -f $(MAIN).pdf

# Watch for changes and rebuild (requires fswatch or entr)
watch:
	@echo "Watching for changes... (Ctrl+C to stop)"
	@while true; do \
		$(MAKE) quick; \
		sleep 2; \
	done

# Create arxiv submission tarball
arxiv: $(MAIN).pdf
	tar -cvf arxiv-submission.tar \
		$(MAIN).tex \
		$(MAIN).bbl \
		appendix.tex \
		figures/autonomy-gradient.tex \
		figures/workflow.tex
	@echo "Created arxiv-submission.tar"
	@echo "Note: Upload the .bbl file, not .bib, to arxiv"

.PHONY: all quick clean distclean watch arxiv
