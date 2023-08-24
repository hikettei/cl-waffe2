
## Ref: https://github.com/ggerganov/ggml/blob/master/examples/gpt-2/convert-ckpt-to-ggml.py

## [TODO] A smol python package to convert ckpt/torch/onnx models into cl-waffe2 model
## [TODO] create a cl-waffe2 format?

## Converts .ckpt file into numpy file

import sys
import json
import struct
import numpy as np
import tensorflow as tf
import os

def assure_dir(path):
    dirname = "/".join(path.split("/")[:-1])
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    return path

if len(sys.argv) < 2:
    print("Usage: convert_model.py dir-model\n")
    sys.exit(1)

# output in the same directory as the model
dir_model = sys.argv[1]

fname_out = sys.argv[1] + "/gpt2-waffe2/"

list_vars = tf.train.list_variables(dir_model)

for name, shape in list_vars:
    print("Processing variable: " + name + " with shape: ", shape)

    data = tf.train.load_variable(dir_model, name)
    n_dims = len(data.shape)

    # for efficiency - transpose the projection matrices
    # "model/h.*/attn/c_attn/w"
    # "model/h.*/attn/c_proj/w"
    # "model/h.*/mlp/c_fc/w"
    # "model/h.*/mlp/c_proj/w"
    if name[-14:] == "/attn/c_attn/w" or \
       name[-14:] == "/attn/c_proj/w" or \
       name[-11:] == "/mlp/c_fc/w" or \
       name[-13:] == "/mlp/c_proj/w":
        print("  Transposing...")
        data = data.transpose()

    dshape = data.shape
    data  = data.astype(np.float32)
    np.save(assure_dir(fname_out + name + ".npy"), data)

print("Done. Output file: " + fname_out)
print("")
