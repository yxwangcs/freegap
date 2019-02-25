#!/bin/sh

mkdir -p datasets
# download kosarak
wget http://fimi.uantwerpen.be/data/kosarak.dat -O datasets/kosarak.dat

# download BMS-POS
wget https://raw.githubusercontent.com/cpearce/HARM/master/datasets/BMS-POS.csv -O datasets/BMS-POS.dat
