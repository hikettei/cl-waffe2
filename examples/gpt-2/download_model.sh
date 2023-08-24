#!/bin/bash

model="117M"

mkdir -p ./examples/gpt-2/assets/models/gpt-2-$model

for file in checkpoint encoder.json hparams.json model.ckpt.data-00000-of-00001 model.ckpt.index model.ckpt.meta vocab.bpe; do
    wget --quiet --show-progress -O ./examples/gpt-2/assets/models/gpt-2-$model/$file https://openaipublic.blob.core.windows.net/gpt-2/models/$model/$file
done

printf "Done! Model '$model' saved in './assets/models/gpt-2-$model/'\n\n"

