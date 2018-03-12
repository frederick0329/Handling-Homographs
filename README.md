### Handling Homographs in Neural Machine Translation 
Thie repo contains implementation of our paper 

[Handling Homographs in Neural Machine Translation](https://arxiv.org/abs/1708.06510)

The code extends [OpenNMT](https://github.com/OpenNMT/OpenNMT) and include variations of source word embedding to tackle the problem of homographs.

Please checkout their github page for the latest information.

### Dependencies

* `nn`
* `nngraph`
* `tds`
* `penlight`

GPU training requires:

* `cunn`
* `cutorch`

Multi-GPU training additionally requires:

* `threads`

## Quickstart

OpenNMT consists of three commands:

1) Preprocess the data.

```th preprocess.lua -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo```

2) Train the model.

```th train.lua -data data/demo-train.t7 -save_model model```

3) Translate sentences.

```th translate.lua -model model_final.t7 -src data/src-test.txt -output pred.txt```
