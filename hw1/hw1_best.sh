#!/bin/sh
python Utils/load_data.py $1
python best_model.py -t -o $2 -m ./model/best_model.ckpt -d $1 -b 16