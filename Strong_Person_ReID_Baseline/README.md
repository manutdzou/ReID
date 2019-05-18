# Strong Person ReID Baseline
The architecture follows the NTU ROSE ReID Project guide line [Person_ReID_Baseline](https://github.com/LinShanify/Person_ReID_Baseline). Some of codes are copy from L1aoXingyu's [reid_baseline](https://github.com/L1aoXingyu/reid_baseline).

* `ResNet50 Last Stride 1` from huanghoujing's [triplet baseline](https://github.com/huanghoujing/person-reid-triplet-loss-baseline) 
* `WarmupMultiStepLR`: is from FAIR's paper: _'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'_


## Requirements
- [python 3](https://www.python.org/downloads/)
- [pytorch 1.0 + torchvision](https://pytorch.org/)
- [yacs](https://github.com/rbgirshick/yacs) Yet Another Configuration System
- [fire](https://github.com/google/python-fire) Automatically generating command line interfaces (CLIs)

Install all dependences libraries
``` bash
pip3 install -r requirements.txt
```

## Configs

Use different yaml config files for different experiment settings. All the config files are store in folder `config`. Please use different `OUTPUT_DIR` names for different experiments to avoid conflit and accidentally files overwritten.


## Datasets
This code support CUHK03, Market1501, DukeMTMC and MSMT17 datasets. All these dataset should be defined in the `DATASETS.NAMES` of the config file, our code will be download the corresponding dataset automatically (into the `datasets` folder). As this fuction require access to __Google Drive__, it will not work in China. 
Currently support:
* [CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)
* [Market1501](http://www.liangzheng.org/Project/project_reid.html)
* [DukeMTMC](https://github.com/layumi/DukeMTMC-reID_evaluation)
* [MSMT17](https://www.pkuvmc.com/publications/msmt17.html)

## Training:
* As the last feature maps increase. Batch Size 128 will use more than 12G
* Batch Size 64 use around 10G GPU memory
``` bash
python train.py ./config/market_softmax.yaml

### Change GPU
python train.py ./config/market_softmax.yaml --DEVICE=cuda:5
```

## Testing:
``` bash
### No Re-Ranking
python test.py ./config/market_softmax.yaml

### Change GPU
python test.py ./config/market_softmax.yaml --DEVICE=cuda:5

### With Re-Ranking
python test.py ./config/market_softmax.yaml --RE_RANKING=True
```

## Testing Cross Dataset:
``` bash
### Market1501 -> DukeMTMC
python test_cross_dataset.py ./config/market_softmax.yaml DukeMTMC
```

## Results Compare with [Person ReID Baseline](https://github.com/LinShanify/Person_ReID_Baseline)
##### Softmax Only Batch Size 64: Rank1 (mAP)

|Dataset     |    Softmax  |Strong Softmax|
|     ---    |     --      | --           |
| CUHK03     | 56.1 (52.4) | 61.1 (56.2   |
| Market1501 | 91.6 (78.7) | 92.5 (80.2)  |
| DukeMTMC   | 83.4 (66.6) | 84.8 (68.3)  |
| MSMT17     | 69.0 (40.1) | 71.4 (42.5)  |

##### Softmax+Tripelt Only Batch Size 64 : Rank1  (mAP)

|            |Softmax+Triplet| Strong Softmax+Triplet |
|     ---    |     --        | --                     |
| CUHK03     | 65.6 (61.8)   | 66.3 (61.8)            |
| Market1501 | 93.2 (82.0)   | 93.4 (83.1)            |
| DukeMTMC   | 86.4 (72.4)   | 86.2 (72.5)            |
| MSMT17     | 73.9 (46.4)   | 74.6 (47.3)            |

