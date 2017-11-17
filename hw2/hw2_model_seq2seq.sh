#!/bin/sh
#python data_preeprocessing.py $1
python model_seq2seq.py -d $1 -o $2 -m ./model_s2s/model1-100 -t