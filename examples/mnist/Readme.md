
## Downloading Assets

```sh
python train_data.py
```

## Usage

```lisp
(load "./cl-waffe2.asd")
(load "./examples/mnist/mnist.asd")
(ql:quickload :mnist-sample)
(mnist-sample::train-and-valid-mlp :epoch-num 10)
```

This process is included in `GNUmakefile`
