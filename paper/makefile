clear : 
	rm -f paper-blx.bib
	rm -f paper.aux
	rm -f paper.bbl
	rm -f paper.blg
	rm -f paper.dvi
	rm -f paper.log
	rm -f paper.pdf
	rm -f paper.run.xml

pdf : paper.tex paper.bib improper-gamma.ps bandwidth-comparison.ps
	latex paper
	bibtex paper
	latex paper
	latex paper
	dvipdf paper