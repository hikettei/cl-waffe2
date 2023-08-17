
(in-package :cl-user)

(defpackage :cl-waffe2/backends.cpu
  (:documentation "The package :cl-waffe2/backends.cpu provides the BLAS Backend to cl-waffe. Note that this package is SBCL-Dependant.")
  (:use :cl :cl-waffe2/vm.generic-tensor :cl-waffe2/vm.nodes :cffi :cl-waffe2/base-impl :cl-waffe2-simd)
  (:export
   :CPUTensor
   :find-and-load-libblas))

(in-package :cl-waffe2/backends.cpu)


(defparameter *openblas-found-p* nil)
(defparameter *simd-extension-p* nil)
(defparameter *one-dnn-found-p* nil) ;; Still not yet available for a while
;; Utils
(eval-when (:compile-toplevel :load-toplevel :execute)

(defun list-to-hash (list)
  (let ((hash (make-hash-table :test 'eql)))
    (dolist (element list)
      (setf (gethash (car element) hash) (cdr element)))
    hash))

(setf cl-waffe2/vm.generic-tensor:*using-backend* `(cl-waffe2/backends.cpu:CPUTensor cl-waffe2/backends.lisp:LispTensor))

;; TODO: Delete this alert with *cl-waffe-never-use-blas* = t
(defun could-not-find ()
  (warn "
cl-waffe2 tried to use OpenBLAS, but CFFI could not find the OpenBLAS shared library (e.g.: libblas.dylib, the path could be found with the locate command or something), therefore, the operations continuing without blas, and using LispTensor.

**`LispTensor` do not provide SIMD-enabled backend and implementation for `MatmulNode`.**

So, if you want to use them, Explict the library path by adding the following code to the initialisation file (e.g.: ~~/.sbclrc, ~~/.roswell/init.lisp), and invoke the (cl-waffe2/backends.cpu:find-and-load-libblas) function. 

For example:
```lisp
;; In ~~/.sbclrc for example:
(defparameter *cl-waffe-config*
    `((:libblas \"libblas.dylib\")))
```
")

  (setf cl-waffe2/vm.generic-tensor:*using-backend* `(cl-waffe2/backends.lisp:LispTensor)))

(cffi:define-foreign-library blas-lib
  (t (:default "libblas")))

(defun find-and-load-libblas ()
  (if (boundp 'cl-user::*cl-waffe-config*)
      (let ((config (list-to-hash (eval 'cl-user::*cl-waffe-config*))))
	(dolist (path (gethash :libblas config))
	  (load-foreign-library (pathname path)))
	(setf *openblas-found-p* t))
      ;; Continue by finding with default path:
      (handler-case (progn
		      (load-foreign-library 'blas-lib)
		      (setf *openblas-found-p* t))
	(error (c)
	  (declare (ignore c))
	  (could-not-find))))
  nil)

(defun warn-blas-without-sbcl ()
  (if (boundp 'cl-user::*cl-waffe-config*)
      (let ((config (list-to-hash (eval 'cl-user::*cl-waffe-config*))))
	(when (gethash :libblas config)
	  (warn "cl-waffe ignored :libblas option because BLAS backend is SBCL-Dependant.")))))
) ;; eval-when

;; Load libblas.dylib
(eval-when (:compile-toplevel :load-toplevel :execute)
  (find-and-load-libblas))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (setf *simd-extension-p* (try-loading-simd-extension)))

