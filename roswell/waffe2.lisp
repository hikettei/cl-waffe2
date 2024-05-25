
(cl:in-package :cl-user)

(progn ;;init forms
  (ros:ensure-asdf)
  #+quicklisp(cl:push (cl:pathname "./") ql:*local-project-directories*)
  #+quicklisp(ql:quickload '(:cl-waffe2 :clingon :rove :cl-ansi-text) :silent t))

(defpackage :waffe2
  (:use :cl :cl-ansi-text)
  (:export :main))

(in-package :waffe2)

(defparameter *use-ansi-color* t)

(defmacro maybe-ansi (op &rest args)
  `(if *use-ansi-color*
       (,op ,@args)
       ,@args))

(defun timestamp ()
  (multiple-value-bind
        (second minute hour day month year day-of-week dst-p tz)
      (get-decoded-time)
    (declare (ignore day-of-week dst-p))

    (maybe-ansi
     blue
     (format nil "[~2,'0d:~2,'0d:~2,'0d, ~d/~2,'0d/~d (GMT~@d)]"
	     hour
	     minute
	     second
	     month
	     day
	     year
	     (- tz)))))

(defun print-info (content)
  (format t "~a : ~a~%" (timestamp) content))

(defun parse-test-config (config)
  (let ((args (uiop:split-string config :separator ",")))
    (loop for arg in args
	  for parsed = (uiop:split-string arg :separator "=")
	  collect (cons (symbol-name (read-from-string (car parsed))) (read-from-string (second parsed))))))

(defun parse-test-config-kwargs (config)
  (let ((args (uiop:split-string config :separator ",")))
    (loop for arg in args
	  for parsed = (uiop:split-string arg :separator "=")
	  append (list (intern (symbol-name (read-from-string (car parsed))) "KEYWORD") (read-from-string (second parsed))))))
#+(or)(print (parse-test-config-kwargs ""))

(defun str->backend (name)
  (let ((available-backends (map 'list #'class-name (alexandria:flatten (cl-waffe2:find-available-backends)))))
    (loop for candidate in available-backends
	  if (equalp (symbol-name candidate) name)
	    do (return-from str->backend candidate))
    (error "Unknown backend: ~a~%Available List: ~a" name available-backends)))

(defun waffe2/demo (cmd)
  (let ((model (or (clingon:getopt cmd :example) "")))
    (macrolet ((of (name)
		 `(equalp model ,name)))
      (cond
	((of "mnist")
	 ;; ros config set dynamic-space-size 4gb
	 (print-info "Loading the example project...")
	 (load "./examples/mnist/mnist.asd")
	 (ql:quickload :mnist-sample :silent t)
	 (print-info "Starting the demonstration...")
	 (apply #'uiop:symbol-call :mnist-sample :train-and-valid-mlp (parse-test-config-kwargs (clingon:getopt cmd :config "")))
	 (print-info "Completed"))
	(T
	 (error "--example ~a is not available." model))))))

(defun waffe2/handler (cmd)
  (let* ((backends (or (clingon:getopt cmd :backends) `("LispTensor")))
	 (*use-ansi-color* (clingon:getopt cmd :ansi-color t)))
    ;; Configure runtimes toplevel
    (apply #'cl-waffe2:set-devices-toplevel (map 'list #'str->backend backends))
    (macrolet ((of (name)
		 `(equalp *mode* ,name)))
      (cond
	((of "test")
	 (print-info "Running the test...")
	 (asdf:load-system :cl-waffe2/test)
	 (uiop:symbol-call :cl-waffe2/tester :running-test :style (intern (string-upcase (clingon:getopt cmd :style "dot")) "KEYWORD"))
	 (print-info "Completed")
	 t)
	((of "gendoc")
	 (print-info "Generating the documents...")
	 (ql:quickload :cl-waffe2/docs :silent t)
	 (uiop:symbol-call :cl-waffe2.docs :generate)
	 (print-info "Completed")
	 t)
	((of "demo")
	 (waffe2/demo cmd))
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
    :key :style)
   (clingon:make-option
    :string
    :description "Example Project to use for demonstration."
    :short-name #\e
    :long-name "example"
    :key :example)
   (clingon:make-option
    :boolean
    :description "Enables/Disables the cl-ansi-color"
    :short-name #\a
    :long-name "ansi-color"
    :key :ansi-color)
   (clingon:make-option
    :string
    :description "Additional configurations. (e.g.: epoch=10,batch=1)"
    :short-name #\c
    :long-name "config"
    :key :config)))

(defun waffe2/command ()
  (clingon:make-command
   :name "cl-waffe2"
   :description "Command Line Tool for cl-waffe2"
   :authors '("hikettei <ichndm@gmail.com>")
   :license "MIT"
   :options (waffe2/options)
   :usage "[ test | gendoc | gencode | demo ] [options]

COMMANDS:
  - test           Tests all principle abstractnode operations work with the provided backends.
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
