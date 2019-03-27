#!/bin/sh

mkdir -p datasets
# download kosarak
wget http://fimi.uantwerpen.be/data/kosarak.dat -O datasets/kosarak.dat

# download BMS-POS
wget https://raw.githubusercontent.com/cpearce/HARM/master/datasets/BMS-POS.csv -O datasets/BMS-POS.dat

# download census data
wget https://www2.census.gov/census_2010/04-Summary_File_1/Pennsylvania/pa2010.sf1.zip -O datasets/pa2010.sf1.zip
unzip datasets/pa2010.sf1.zip -d datasets/pa2010

# download T40I10D100K
wget http://fimi.uantwerpen.be/data/T40I10D100K.dat -O datasets/T40I10D100K.dat