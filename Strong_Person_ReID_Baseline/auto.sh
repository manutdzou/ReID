#!/bin/sh

python test_cross_dataset.py ./config/market_softmax_triplet_colorjitter.yaml DukeMTMC --DEVICE=cuda:7
python test_cross_dataset.py ./config/market_softmax_triplet_colorjitter.yaml NTUCampus --DEVICE=cuda:7
python test_cross_dataset.py ./config/market_softmax_triplet.yaml DukeMTMC --DEVICE=cuda:7
python test_cross_dataset.py ./config/market_softmax_triplet.yaml NTUCampus --DEVICE=cuda:7

python test_cross_dataset.py ./config/duke_softmax_triplet_colorjitter.yaml Market1501 --DEVICE=cuda:7
python test_cross_dataset.py ./config/duke_softmax_triplet_colorjitter.yaml NTUCampus --DEVICE=cuda:7
python test_cross_dataset.py ./config/duke_softmax_triplet.yaml Market1501 --DEVICE=cuda:7
python test_cross_dataset.py ./config/duke_softmax_triplet.yaml NTUCampus --DEVICE=cuda:7

