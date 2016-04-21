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

ejs : lyx body.tex abstract.tex ejs.tex paper.bib improper-gamma.pdf bandwidth-comparison.pdf
	pdflatex ejs
	bibtex ejs
	pdflatex ejs
	pdflatex ejs
	open ejs.pdf


lyx : wrapper.lyx body.lyx abstract.lyx
	lyx --force-overwrite --export pdflatex wrapper.lyx
	rm wrapper.tex

all : clear lyx ejs