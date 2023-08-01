
(in-package :cl-user)

(defpackage :cl-waffe2.docs
  (:use
   :cl
   :cl-waffe2
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes
   :cl-waffe2/distributions
   :cl-waffe2/base-impl
   :cl-waffe2/backends.lisp
   :cl-waffe2/backends.cpu
   :cl-waffe2/backends.jit.cpu
   :cl-waffe2/nn
   :cl-waffe2/optimizers
   :cl-ppcre)
  (:export
   #:generate))

(in-package :cl-waffe2.docs)

;; Utils for generating markdown

(defparameter *page-out-to* nil)
(defparameter *nest-n* 2)

(defmacro with-page (title-binding-symbol
		     title-name
		     &body
		       body)
  `(setq ,title-binding-symbol
	 (with-output-to-string (*page-out-to*)
	   (with-top ,title-name
	     ,@body))))

(defmacro with-section (title-name &body body)
  `(progn
     (format *page-out-to* "~%~a ~a~%"
	     ,(with-output-to-string (o) (dotimes (i *nest-n*) (princ "#" o)))
	     ,title-name)
     (let ((*nest-n* (1+ *nest-n*)))
       ,@body)))

(defmacro with-top (title-name &body body)
  `(progn
     (format *page-out-to* "~%# ~a~%" ,title-name)
     ,@body))

(defmacro insert (content &rest args)
  `(format *page-out-to* ,content ,@args))

(defun placedoc (object type)
  (when (eql type 'macro)
    (return-from placedoc (placedoc (macro-function object) 'function)))
  
  (let ((docstring (documentation object type)))
    (format *page-out-to* "~a" docstring)))

(defun with-example (source)
  (let ((result (eval (read-from-string source))))
    (format *page-out-to*
	    "
### Example

```lisp
~a

~a
```
"
	    source result)))

(defun with-examples (&rest codes)
  (format *page-out-to* "~%###Example")
  (dolist (c codes)
    (format *page-out-to* "~%**REPL:**~%```lisp~%> ~a~%```" c)
    (let ((result (eval (read-from-string c))))
      (format *page-out-to* "~%```~%~a~%```~%" result))))

(defparameter *target-dir* "./docs/cl-waffe2-docs/docs")

(defun write-scr (filepath content)
  (with-open-file (str (out-dir filepath)
                       :direction :output
                       :if-exists :supersede
                       :if-does-not-exist :create)
    (format str "~a" content)))

(defun out-dir (name)
  (format nil "~a/~a.md" *target-dir* name))

(defparameter *distributions* "")
(defparameter *generic-tensor* "")
(defparameter *nodes* "")
(defparameter *base-impl* "")
(defparameter *base-impl-nodes* "")

(defparameter *nn* "")
(defparameter *optimizer* "")

(defparameter *cl-waffe2-package* "")

(defparameter *lisp-tensor-backend* "")
(defparameter *cpu-tensor-backend* "")
(defparameter *cpu-jit-tensor-backend* "")

(defun generate ()
  (write-scr "generic-tensor" *generic-tensor*)
  (write-scr "base-impl" *base-impl*)
  (write-scr "base-impl-nodes" *base-impl-nodes*)
  (write-scr "nodes" *nodes*)
  (write-scr "distributions" *distributions*)

  (write-scr "nn" *nn*)
  (write-scr "optimizer" *optimizer*)

  (write-scr "utils" *cl-waffe2-package*)

  (write-scr "lisp-tensor-backend"     *lisp-tensor-backend*)
  (write-scr "cpu-tensor-backend"      *cpu-tensor-backend*)
  (write-scr "cpu-jit-tensor-backend"  *cpu-jit-tensor-backend*)

  (format t "Completed~%")
  )
