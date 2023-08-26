#!/bin/bash

model="117M"

mkdir -p ./examples/gpt-2/assets/models/gpt-2-$model

# 
for file in checkpoint encoder.json model.ckpt.data-00000-of-00001 model.ckpt.index model.ckpt.meta hparams.json vocab.bpe; do
    wget --quiet --show-progress -O ./examples/gpt-2/assets/models/gpt-2-$model/$file https://openaipublic.blob.core.windows.net/gpt-2/models/$model/$file
done

printf "Done! Model '$model' saved in './assets/models/gpt-2-$model/'\n\n"

#curl --output ./examples/gpt-2/assets/gpt2-pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin

