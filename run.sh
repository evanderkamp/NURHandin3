#!/bin/bash

echo "Run handin 3 Evelyn van der Kamp s2138085"

echo "Download txt file to use in script"
if [ ! -e satgals_m11.txt ]; then
  wget home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m11.txt
fi

if [ ! -e satgals_m12.txt ]; then
  wget home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m12.txt
fi

if [ ! -e satgals_m13.txt ]; then
  wget home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m13.txt
fi

if [ ! -e satgals_m14.txt ]; then
  wget home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m14.txt
fi

if [ ! -e satgals_m15.txt ]; then
  wget home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m15.txt
fi

# First exercise
echo "Run the first script ..."
python3 NUR_handin3Q1.py



echo "Generating the pdf"

pdflatex Handin3.tex
bibtex Handin3.aux
pdflatex Handin3.tex
pdflatex Handin3.tex
