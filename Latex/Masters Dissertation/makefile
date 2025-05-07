# You want latexmk to *always* run, because make does not have all the info.
# Also, include non-file targets in .PHONY so they are run regardless of any
# file of the given name existing.

rootFile = main.tex
outFile = main.pdf

.PHONY: default all main.pdf font_embed clean

# Final targets
default: main.pdf

all: main.pdf font_embed

# Compilation
main.pdf: main.tex
	@latexmk -halt-on-error -output-directory=build -synctex=1 -lualatex $<
	@cp build/$(basename $^).pdf $@

# Font embedding rule
font_embed: main.pdf
	rm $<
	gs -q -dNOPAUSE -dBATCH -dPrinted=false -dPDFSETTINGS=/prepress -sDEVICE=pdfwrite -sOutputFile=$< build/main.pdf

# Cleanup
clean:
	@rm -rf build/
	@rm -f $(outFile) $(fontEmbeddedOutFile)
	@echo "Clean complete"
