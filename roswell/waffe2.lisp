
(cl:in-package :cl-user)

(progn ;;init forms
  (ros:ensure-asdf)
  #+quicklisp(cl:push (cl:pathname "./") ql:*local-project-directories*)
  #+quicklisp(ql:quickload '(:cl-waffe2 :clingon) :silent t))

(defpackage :waffe2
  (:use :cl)
  (:export :main))

(in-package :waffe2)

(defun parse-test-config (config)
  (let ((args (uiop:split-string config :separator ",")))
    (loop for arg in args
	  for parsed = (uiop:split-string arg :separator "=")
	  collect (cons (symbol-name (read-from-string (car parsed))) (read-from-string (second parsed))))))
#+(or)(print (parse-test-config "M=1, K=2"))

(defun str->backend (name)
  (let ((available-backends (map 'list #'class-name (alexandria:flatten (cl-waffe2:find-available-backends)))))
    (loop for candidate in available-backends
	  if (equalp (symbol-name candidate) name)
	    do (return-from str->backend candidate))
    (error "Unknown backend: ~a~%Available List: ~a" name available-backends)))

(defun waffe2/handler (cmd)
  (let* ((backends (or (clingon:getopt cmd :backends) `("LispTensor"))))
    ;; Configure runtimes toplevel
    (apply #'cl-waffe2:set-devices-toplevel (map 'list #'str->backend backends))
    (macrolet ((of (name)
		 `(string= *mode* ,name)))
      (cond
	((of "test")
	 (asdf:load-system :cl-waffe2/test)
	 (uiop:symbol-call :cl-waffe2/tester :running-test :style (intern (string-upcase (clingon:getopt cmd :style "dot")) "KEYWORD"))
	 t)
	((of "gendoc")
	 (ql:quickload :cl-waffe2/docs :silent t)
	 (uiop:symbol-call :cl-waffe2.docs :generate)
	 t)
	((of "gencode")
	 ;; WIP From ONNX -> C/C++/CUDA Code Generator + Mimimun C Interpreter
	 (error "Export2Clang Mode is not yet ready.")
	 t)
	(T nil)))))

(defun waffe2/options ()
  (list
   (clingon:make-option
    :list
    :description "Backend to use (e.g.: $ waffe2 test -b CPUTensor -b LispTensor ...)"
    :short-name #\b
    :long-name "backend"
    :key :backends)
   (clingon:make-option
    :string
    :description "Style for the testing tool. One of dot, spec, none. (conform to rove)"
    :short-name #\s
    :long-name "style"
    :key :style)))

(defun waffe2/command ()
  (clingon:make-command
   :name "cl-waffe2"
   :description "Command Line Tool for cl-waffe2"
   :authors '("hikettei <ichndm@gmail.com>")
   :license "MIT"
   :options (waffe2/options)
   :usage "[ unittest | test | gendoc | gencode ] [options]

COMMANDS:
  - unittest       Tests if all principle abstractnode works given backends.
  - test           Run all tests under the given backends.
  - test[package]  Run all tests
  - gendoc         Generates the cl-waffe2 documentation.
  - gencode        [WIP] Mimimum C runtime generator (from ONNX)"
   :handler #'waffe2/handler))

(defparameter *mode* "")
(defun main (&rest argv)
  (let ((app (waffe2/command)))
    (if (= (length argv) 0)
	(clingon:print-usage app t)
	(let ((*mode* (car argv)))
	  (or (clingon:run app argv) (clingon:print-usage app t))))))
