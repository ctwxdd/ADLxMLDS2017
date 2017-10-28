#!/bin/sh
python Utils/load_data.py $1
python model_rnn.py -t -o $2 -m ./model/rnn_model.ckpt -d ./ -b 16