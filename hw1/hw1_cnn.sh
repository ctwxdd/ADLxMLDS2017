#!/bin/sh
python Utils/load_data.py $1
python model_cnn.py -t -o $2 -m ./model/cnn_model.ckpt -d $1 -b 16