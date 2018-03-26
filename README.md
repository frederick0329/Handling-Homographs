# Handling Homographs in Neural Machine Translation 
This repo is currenlty work in progress.

This repo contains implementation of our paper 

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

```th preprocess.lua -train_src wmt_en_de/train.en -train_tgt wmt_en_de/train.de -valid_src wmt_en_de/newstest2013.en -valid_tgt wmt_en_de/newstest2013.de -save_data wmt_en_de/preprocessed```

2) Train the model.

The command for the following setting
- -gate : use the gating mechanism to integrate context vector and word embedding
- -gating_type cbow : use nbow(cbow) as context network type
- -share : force the input embedding for the context network to share the parameters with the input for the main rnn
- -brnn : use bi-rnn as the main rnn

```th train.lua -data wmt_en_de/preprocessed_ende-train.t7 -save_model model -gate -gating_type cbow -share -brnn -gpuid 1```

The command for the following setting
- -concat : use the concat mechanism to integrate context vector and word embedding
- -gating_type contextBiEncoder : use a bi-encoder as context network type
- -share : force the input embedding for the context network to share the parameters with the input for the main rnn
- -brnn : use bi-rnn as the main rnn

```th train.lua -data wmt_en_de/preprocessed_ende-train.t7 -save_model model -concat -gating_type contextBiEncoder -share -brnn -gpuid 1```

3) Translate sentences.

The command for the following setting
- -gate : use the gating mechanism to integrate context vector and word embedding
- -gating_type cbow : use nbow(cbow) as context network type

```th translate.lua -gate -gating_type cbow -model model_final.t7 -src wmt_en_de/newstest2014.en -tgt wmt_en_de/newstest2014.de -output pred.txt -replace_unk -gpuid 1```

4) Note

The input options must match for the contextNetwork during the training process and the translating process.
