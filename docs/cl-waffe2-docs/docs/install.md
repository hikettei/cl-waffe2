
# Setting up Environments

## Are you new to Common Lisp?

I know... there are few people who are attempted to do ML/DL in Common Lisp! while other languages provide a strong baseline and good platforms. However, I still believe in the clear benefits of doing such tasks on Common Lisp.

I've been working with Common Lisp for the past two years, but I realise the attraction that no other language can replace it. At first glance, indeed this language has a strange syntax, and some features of the language may seem too much. but believe me! One day you will learn to use it. In fact, I guess cl-waffe2 is portable to ANSI Common Lisp, but not portable to non-lisp languages.

Anyway, the first step is to set up Common Lisp Environment.

I don't know which is best, and this is just my recommendations.

### 1. Installing Roswell

Roswell is environment manager of Common Lisp (and much more!)

<https://github.com/roswell/roswell>

See the `Readme.md` and install Roswell

### 2. Installing SBCL

There are many implementations of Common Lisp, and SBCL is one of the processing system.

As of this writing(2023/07/02), some features of cl-waffe2 are SBCL-dependent, so this one is recommended.

With roswell:

```sh
$ ros install sbcl
$ ros use <Installed SBCL Version>
$ ros run # REPL is launched.
```

should work.

### 3. Setting up IDE (Optional)

I guess It's a pity to write Common Lisp without REPL. There are a lot of options, but as far as I know, `Emacs with SLIME` or `Lem` is widely supported choice.


## Installing cl-waffe2

With roswell, the latest repository can be fetched which is also recognised by `quicklisp`

```sh
$ ros install hikettei/cl-waffe2
```

Another option is to load `cl-waffe2.asd` configurations manually after cloning cl-waffe2 github repos.

```sh
$ git clone <Repository>
$ cd ./cl-waffe2
$ ros run # start repl
$ (load "cl-waffe2.asd")
$ (ql:quickload :cl-waffe2)
$ (in-pacakge :cl-waffe2-repl)
```


With quicklisp:

(It's going to take a while...)

## OpenBLAS Backend

In your init file, (e.g.: `~/.roswell/init.lisp` or `~/.sbclrc`), add the code below for example. (change the path depending on your environment).

```lisp
;; In ~~/.sbclrc for example:
(defparameter *cl-waffe-config*
    `((:libblas \"libblas.dylib for example\")))
```

One of cl-waffe2 backends `CPUTensor` loads the OpenBLAS shared library of the path written in the `cl-user::*cl-waffe-config*` parameter when cl-waffe2 loaded.

## CUDA Backend

(Currently not supported yet...)

