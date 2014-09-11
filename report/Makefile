# Arquivo Makefile para automaticamente chamar o BiBTeX e o LaTeX
# no número de vezes mais comumente utilizado.  Se você desejar
# algo mais robusto, posso recomendar o latexmk [1].
#
# Para usar, basta executar:
#
#     make
#
# dentro do diretório da report.  Se o LaTeX estiver
# emperrado, você pode tentar limpar seus arquivos auxiliares
# usando:
#
#     make clean
#
# [1] http://www.phys.psu.edu/~collins/software/latexmk/

default: pdf

pdf:
	pdflatex report.tex
	bibtex report
	pdflatex report.tex
	pdflatex report.tex

ps: dvi
	dvips report.dvi

dvi:
	latex report.tex
	bibtex report
	latex report.tex
	latex report.tex

clean:
	rm -vf report.{aux,bbl,blg,brf,fdb_latexmk,lof,log,lot,out,toc}

mrproper: clean
	rm -vf report.{dvi,pdf,ps}
