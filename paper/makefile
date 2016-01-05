clear :
	rm -f paper.tex
	rm -f paper-blx.bib
	rm -f paper.aux
	rm -f paper.bbl
	rm -f paper.blg
	rm -f paper.dvi
	rm -f paper.log
	rm -f paper.pdf
	rm -f paper.run.xml

latex : paper.tex paper.bib improper-gamma.ps bandwidth-comparison.ps
	latex paper
	bibtex paper
	latex paper
	latex paper
	dvipdf paper
	open paper.pdf

pdflatex: paper.tex paper.bib improper-gamma.pdf bandwidth-comparison.pdf
	pdflatex paper
	bibtex paper
	pdflatex paper
	pdflatex paper
	open paper.pdf

lyx : paper.lyx
	lyx --force-overwrite --export pdflatex paper.lyx

all : clear lyx pdflatex