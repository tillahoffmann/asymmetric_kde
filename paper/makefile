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

statsoc : paper.tex master_statsoc.tex paper.bib improper-gamma.ps bandwidth-comparison.ps
	latex master_statsoc
	bibtex master_statsoc
	latex master_statsoc
	latex master_statsoc
	dvipdf master_statsoc
	open master_statsoc.pdf


lyx : paper.lyx
	lyx --force-overwrite --export pdflatex master.lyx
	rm master.tex

all : clear lyx pdflatex