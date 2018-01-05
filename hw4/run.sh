#!/bin/sh
wget https://www.dropbox.com/s/v43blldpyg7yx7f/model-350500.data-00000-of-00001?dl=1 -P ./ckpt/ -O model-350500.data-00000-of-00001
wget https://www.dropbox.com/s/wupexxm1986v1ah/model-350500.index?dl=1 -P ./ckpt/ -O model-350500.index
wget https://www.dropbox.com/s/04clowwjp5c4y1t/model-350500.meta?dl=1 -P ./ckpt/ -O model-350500.meta
python generate.py -t $1