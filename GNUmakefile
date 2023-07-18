SBCL                := sbcl
SBCL_OPTIONS        := --noinform
QUICKLOAD_WAFFE2    := --load cl-waffe2.asd --eval '(ql:quickload :cl-waffe2)'
MKTEMP              := mktemp
RLWRAP              := rlwrap
LOGFILE             := $(shell mktemp)

.PHONY: all compile test recordtest repl rlrepl record docs clean delete_quicklisp install_quicklisp add_to_init_file

all:
	$(info To compile, type `make compile`)
	$(info To test, type `make test`)
	$(info To record test, type `make recordtest`)
	$(info To run REPL, type `make repl`)
	$(info To run REPL with rlwrap, type `make rlrepl`)
	$(info To record REPL session, type `make record`)
	$(info To generate documents, type `make docs`)
	$(info To clean FASL, type `make clean`)

compile:
	$(SBCL) $(SBCL_OPTIONS) $(QUICKLOAD_WAFFE2) \
		--eval '(asdf:compile-system :cl-waffe2)' \
		--quit

test:
	$(SBCL) $(SBCL_OPTIONS) $(QUICKLOAD_WAFFE2) \
		--eval '(asdf:test-system :cl-waffe2)' \
		--quit

recordtest:
	$(warning This session will be recorded in $(LOGFILE))
	$(RLWRAP) --logfile $(LOGFILE) \
	$(SBCL) $(SBCL_OPTIONS) $(QUICKLOAD_WAFFE2) \
		--eval '(asdf:test-system :cl-waffe2)' \
		--quit
	@printf 'This session has been recorded in %s\n' $(LOGFILE)

repl:
	$(SBCL) $(SBCL_OPTIONS) $(QUICKLOAD_WAFFE2)

rlrepl:
	$(RLWRAP) $(SBCL) $(SBCL_OPTIONS) $(QUICKLOAD_WAFFE2)

record:
	$(warning This session will be recorded in $(LOGFILE))
	$(RLWRAP) --logfile $(LOGFILE) \
		$(SBCL) $(SBCL_OPTIONS) $(QUICKLOAD_WAFFE2)
	@printf 'This session has been recorded in %s\n' $(LOGFILE)

docs:
	$(SBCL) $(SBCL_OPTIONS) $(QUICKLOAD_WAFFE2) \
		--eval '(ql:quickload :cl-waffe2/docs)' \
		--eval '(cl-waffe2.docs:generate)' \
		--quit

clean:
	rm -rv ~/.cache/common-lisp/sbcl-*

delete_quicklisp:
	rm -rv ~/quicklisp

TEMP_QUICKLISP := $(shell mktemp)

install_quicklisp:
	curl -sSL -o $(TEMP_QUICKLISP) 'https://beta.quicklisp.org/quicklisp.lisp'
	$(SBCL) $(SBCL_OPTIONS) \
		--load $(TEMP_QUICKLISP) \
		--eval '(quicklisp-quickstart:install)' \
		--quit

add_to_init_file:
	$(SBCL) $(SBCL_OPTIONS) \
		--load ~/quicklisp/setup.lisp \
		--eval '(ql:add-to-init-file)' \
		--quit
