
# Setting up Environments

## If you're new to Common Lisp:

### 1. Installing Roswell

Roswell is an environment manager for Common Lisp.

<https://github.com/roswell/roswell>

See the `Readme.md` and install Roswell

### 2. Installing Common Lisp

Common Lisp has several implementations, but I personally recommended SBCL for its performance.

If you've installed Roswell:

```sh
$ ros install sbcl
$ ros use <Installed SBCL Version>
$ ros run # REPL is launched.
```

should work and everything is done.

### 3. Setting up IDE (optional)

The following editors are recommended as we're working with REPL:

- [Emacs](https://www.gnu.org/software/emacs/) + [Slime](https://slime.common-lisp.dev/)

- [Lem](https://github.com/lem-project/lem) Lem is an emacs-like text editor specialized on Common Lisp.


## Installing cl-waffe2

As of this writing(2023/9/13), cl-waffe2 is not yet available on Quicklisp. So I'm sorry but you have to install it manually.

With roswell, the latest cl-wafe2 repository can be fetch like:

```sh
$ ros install hikettei/cl-waffe2
```

In this case, you have to note that SBCL also needs to be started via Roswell.

Another valid option would be loading `cl-waffe2.asd` file manually after cloning cl-waffe2 github repos:

```sh
$ git clone https://github.com/hikettei/cl-waffe2.git
$ cd ./cl-waffe2
$ ros run # start repl
$ (load "cl-waffe2.asd")
$ (ql:quickload :cl-waffe2)
$ (in-pacakge :cl-waffe2-repl) # or make repl
```

After you ensured it should work, move the `./cl-waffe2` directory to `~/quicklisp/local-projects/` and quicklisp can find the project!

The get the full performance of cl-waffe2, you also have to do the following steps:

### Setting BLAS

cl-waffe2 searches for and reads the `libblas` file by default. The following steps are only necessary if you get a warning when loading the library

First, install the libopenblas

```sh
# with ubuntu for example
$ apt install libopenblas

# With macOS
$ brew instlal libopenblas
```

Load the package again:

```lisp
$ ros run
$ (load "cl-waffe2.asd")
$ (ql:quickload :cl-waffe2)
```

If you've got no warning after loading cl-waffe2, `CPUTensor` is successfully enabled and can recognize the OpenBLAS. If you still get warnings, you have to step an additional configs because cl-waffe2 couldn't find out the location.

So, In your init file, (e.g.: `~/.roswell/init.lisp` or `~/.sbclrc`), add the code below for example. (Change the path depending on your environment. You can find where you've installed the library with `$ locate libblas` for example of macOS).

```lisp
;; In ~~/.sbclrc for example:
(defparameter *cl-waffe-config*
    `((:libblas \"libblas.dylib for example\")))
```

It should work. If you still get warnings or encountered some problems, feel free to make an [issue](https://github.com/hikettei/cl-waffe2/issues).

### Building SIMD Extension

SIMD Extension is an extension for speeding up the execution of mathematical functions and some instructions (including sparse matrix) on the CPU.

```lisp
$ make build_simd_extension
```

and everything is ok. Ensure that no warnings are displayed in your terminal after loaded cl-waffe2.

### Is GPU(CUDA/Metal/OpenCL etc...) supported?

Currently, No. But cl-waffe2 is designed to be independent of which devices work on, and writing extension is easy. Personally, I don't have enough environment and equipment to do the test, so I plan to do it one day when I save up the money.

