## Examples

- `./tutorial_jp.lisp` Learn how to use cl-waffe2 with some use cases. (Japanese only)
- `./mlp_sin_wave.lisp` Minimum configuration of inferencing and model optimisation.
- `./mnist` Trains MNIST Dataset with various models
  - `MLP`+`Adam` -> `./mnist/mlp.lisp`
  - `CNN`+`Adam` -> `./mnist/cnn.lisp`

- `./cifar-10` (TODO)

- `./gpt2` demonstrates how to inference the model with cl-waffe2. (Still not working)

(More things will be added in the future...)

### Makefile

cl-waffe2 requires following things to convert the model format and training data:

- `Python 3.X`
- `Numpy`
- `PyTorch`
- `Tensorflow`

```sh
$ make download_asset  # downloads all assets
$ make example_mnist   # Starts MNIST Example
$ make example_gpt     # Starts GPT2 Inference Example (Still not yet stable)
```
