# Examples

### Tutorials

- [./tutorial_jp.lisp](./tutorial_jp.lisp) Learn how to use cl-waffe2 with some use cases. (Japanese only)
- [./mlp_sin_wave.lisp](./mlp_sin_wave.lisp) Minimum configuration of inferencing and model optimisation.

### Model Zoo

- [./mnist](./mnist) Train and validate MNIST Dataset with various models
  - [MLP and Adam](./mnist/mlp.lisp)
  - [CNN and Adam](./mnist/cnn.lisp)

- [./cifar-10](./cifar-10) (TODO)

- [./gpt2](./gpt-2) demonstrates how to inference GPT2-Model `(GPT2-117M)` with cl-waffe2. (NOT WORKING WELL)

### Notebooks

- (TODO) step-by-step examples with jupyter notebook format...

(More things will be added in the future...)

### Makefile

Some of routines are included in Makefile. cl-waffe2 requires following things to convert the model format and training data:

- `Python 3.X`
- `Numpy`
- `PyTorch`
- `Tensorflow`

```sh
$ make download_asset  # downloads all assets
$ make example_mnist   # Starts MNIST Example
$ make example_gpt     # Starts GPT2 Inference Example (Still not yet stable)
```
