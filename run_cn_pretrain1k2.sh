#!/bin/bash

stage=1

if [ $stage -le 0 ]; then
    python3 examples/wav2vec/wav2vec_manifest.py /NAS5/speech/data/speech/chnl1/pretrain_10k/wav/train \
        --dest /NAS5/speech/data/speech/chnl1/pretrain_10k/manifest/train \
        --ext wav --valid-percent 0
    python3 examples/wav2vec/wav2vec_manifest.py /NAS5/speech/data/speech/chnl1/pretrain_10k/wav/valid \
        --dest /NAS5/speech/data/speech/chnl1/pretrain_10k/manifest/valid \
        --ext wav --valid-percent 0
fi

if [ $stage -le 1 ]; then
    fairseq-hydra-train \
        task.data=/NAS5/speech/data/speech/chnl1/pretrain_10k/manifest/train \
        distributed_training.distributed_world_size=8 \
        +optimization.update_freq='[8]' \
        --config-dir examples/wav2vec/config/pretraining \
        --config-name wav2vec2_base_cn2 
fi