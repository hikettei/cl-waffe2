SBCL                := sbcl
# Avoid loading system-wide libraries (--no-sysinit)
SBCL_OPTIONS        := --noinform --no-sysinit
QUICKLOAD_WAFFE2    := --load cl-waffe2.asd --eval '(ql:quickload :cl-waffe2)'
MKTEMP              := mktemp
RLWRAP              := rlwrap
LOGFILE             := $(shell mktemp)

PYTHON              := "python"
GCC                 := "gcc"
CMAKE               := "cmake"
.DEFAULT_GOAL := help

# This code taken from https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
# Japanese version: https://postd.cc/auto-documented-makefile/
.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: compile
compile: ## Compiles the whole project
	$(SBCL) $(SBCL_OPTIONS) $(QUICKLOAD_WAFFE2) \
		--eval '(asdf:compile-system :cl-waffe2)' \
		--quit

.PHONY: test
test: ## Running a test harness
	$(SBCL) $(SBCL_OPTIONS) $(QUICKLOAD_WAFFE2) \
		--eval '(asdf:test-system :cl-waffe2)' \
		--quit

.PHONY: recordtest
recordtest: ## Running a test harness with recording logs
	$(warning This session will be recorded in $(LOGFILE))
	$(RLWRAP) --logfile $(LOGFILE) \
	$(SBCL) $(SBCL_OPTIONS) $(QUICKLOAD_WAFFE2) \
		--eval '(asdf:test-system :cl-waffe2)' \
		--quit
	@printf 'This session has been recorded in %s\n' $(LOGFILE)

.PHONY: repl
repl: ## Launch REPL with loading cl-waffe2
	$(SBCL) $(SBCL_OPTIONS) $(QUICKLOAD_WAFFE2)

.PHONY: rlrepl
rlrepl: ## Launch REPL with rlwrap
	$(RLWRAP) $(SBCL) $(SBCL_OPTIONS) $(QUICKLOAD_WAFFE2)

.PHONY: record
record: ## Launch REPL with logging
	$(warning This session will be recorded in $(LOGFILE))
	$(RLWRAP) --logfile $(LOGFILE) \
		$(SBCL) $(SBCL_OPTIONS) $(QUICKLOAD_WAFFE2)
	@printf 'This session has been recorded in %s\n' $(LOGFILE)

.PHONY: slynk
slynk: ## Launch Slynk server
	$(SBCL) $(SBCL_OPTIONS) $(QUICKLOAD_WAFFE2) \
		--eval '(ql:quickload :slynk)' \
		--eval '(slynk:create-server)'

.PHONY: swank
swank: ## Launch Swank server
	$(SBCL) $(SBCL_OPTIONS) $(QUICKLOAD_WAFFE2) \
		--eval '(ql:quickload :swank)' \
		--eval '(swank:create-server)'

.PHONY: docs
docs: ## Generate documents
	$(SBCL) $(SBCL_OPTIONS) $(QUICKLOAD_WAFFE2) \
		--eval '(ql:quickload :cl-waffe2/docs)' \
		--eval '(cl-waffe2.docs:generate)' \
		--quit

.PHONY: mkdocs-serve
mkdocs-serve: ## Launchs the documentation server.
	cd ./docs/cl-waffe2-docs && mkdocs serve

.PHONY: rt
rt: recordtest ## Alias for recordtest

.PHONY: rr
rr: rlrepl ## Alias for rlrepl

.PHONY: rd
rd: record ## Alias for record

.PHONY: clean
clean:
	rm -rv ~/.cache/common-lisp/sbcl-*

.PHONY: delete_quicklisp
delete_quicklisp:
	rm -rv ~/quicklisp

TEMP_QUICKLISP := $(shell mktemp)
QUICKLISP_SIGNING_KEY := D7A3489DDEFE32B7D0E7CC61307965AB028B5FF7

.PHONY: install_quicklisp
install_quicklisp: ## Install Quicklisp
	curl -sSL -o $(TEMP_QUICKLISP) 'https://beta.quicklisp.org/quicklisp.lisp'
	curl -sSL -o $(TEMP_QUICKLISP).asc 'https://beta.quicklisp.org/quicklisp.lisp.asc'
	gpg --batch --recv-keys ${QUICKLISP_SIGNING_KEY}
	gpg --batch --verify $(TEMP_QUICKLISP).asc $(TEMP_QUICKLISP)
	$(SBCL) $(SBCL_OPTIONS) \
		--non-interactive \
		--load $(TEMP_QUICKLISP) \
		--eval '(quicklisp-quickstart:install)'

.PHONY: add_to_init_file
add_to_init_file: ## Enable Quicklisp autoloading
	$(SBCL) $(SBCL_OPTIONS) \
		--non-interactive \
		--load ~/quicklisp/setup.lisp \
		--eval '(ql-util:without-prompting (ql:add-to-init-file))'

.PHONY: build_sleef
build_sleef: ## Builds SLEEF
	rm -rf ./third_party/sleef/build && mkdir ./third_party/sleef/build && cd ./third_party/sleef/build &&\
	$(CMAKE) -DISABLE_FLOAT128=true -DISABLE_LONG_DOUBLE=true -DDISABLE_MPFR=TRUE .. &&\
	make && make install

.PHONY: build_simd_extension
build_simd_extension: build_sleef ## Installs SIMD Extension shared library for the CPUTensor backend. (make sure that build_sleef must be done in advance)
	$(GCC) -O3 -march=native -shared -o \
		./source/backends/cpu/cl-waffe2-simd/kernels/cl-waffe2-simd.so \
		-fpic ./source/backends/cpu/cl-waffe2-simd/kernels/cl-waffe2-simd.c -lm -lsleef \

.PHONY: delete_simd_extension
delete_simd_extension: ./source/backends/cpu/cl-waffe2-simd/kernels/cl-waffe2-simd.so ## Deletes Compiled SIMD Extension shared library so that CPUTensor works under OpenBLAS
	rm -rf ./source/backends/cpu/cl-waffe2-simd/kernels/cl-waffe2-simd.so

.PHONY: download_assets
download_assets: ## Downloads training data sample codes use.
	sh ./examples/gpt-2/download_model.sh
	$(PYTHON) ./examples/gpt-2/convert_model.py ./examples/gpt-2/assets/models/gpt-2-117M
	cd ./examples/mnist && $(PYTHON) train_data.py

.PHONY: example_mnist
example_mnist: ## Start MNIST Example and Benchmarking. (download_assets must be called in advance)
	$(SBCL) $(SBCL_OPTIONS) \
		--load ./cl-waffe2.asd --load ./examples/mnist/mnist.asd \
		--eval '(ql:quickload :mnist-sample)' \
		--eval '(mnist-sample::train-and-valid-mlp :epoch-num 10)'

