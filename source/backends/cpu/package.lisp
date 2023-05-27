
(in-package :cl-user)

(defpackage :cl-waffe2/backends.cpu
  (:documentation "The package :cl-waffe2/backends.cpu provides the BLAS Backend to cl-waffe. Note that this package is SBCL-Dependant.")
  (:use :cl :cl-waffe2/vm.generic-tensor :cl-waffe2/vm.nodes :cffi :cl-waffe2/base-impl)
  (:export
   :CPUTensor))

(in-package :cl-waffe2/backends.cpu)


;; Utils
(eval-when (:compile-toplevel :load-toplevel :execute)
  
(defun list-to-hash (list)
  (let ((hash (make-hash-table :test 'eql)))
    (dolist (element list)
      (setf (gethash (car element) hash) (cdr element)))
    hash))

#+sbcl(setf cl-waffe2/vm.generic-tensor:*using-backend* `(cl-waffe2/backends.cpu:CPUTensor))

;; TODO: Delete this alert with *cl-waffe-never-use-blas* = t
(defun could-not-find ()
  (format t "
== Note [About CPUTensor's Performance] ========================
cl-waffe could not find the OpenBLAS shared library (e.g.: libblas.dylib, the path could be found with the locate command or something), continuing without blas, but using the Common Lisp backend. (i.e.:LispTensor)

To continue with BLAS, Add the following code to the initialisation file and restart your program. (e.g.: ~~/.sbclrc, ~~/.roswell/init.lisp), otherwise **THE PERFORMANCE MAY DECREASE**.

```lisp

;; In ~~/.sbclrc for example:
(defparameter *cl-waffe-config*
    `((:libblas \"libblas.dylib for example\")))

```
================================================================
")

  (setf cl-waffe2/vm.generic-tensor:*using-backend* `(cl-waffe2/backends.lisp:LispTensor)))

(defun find-and-load-libblas ()
  (if (boundp 'cl-user::*cl-waffe-config*)
      (let ((config (list-to-hash (eval 'cl-user::*cl-waffe-config*))))
	(dolist (path (gethash :libblas config))
	  (load-foreign-library
	   (pathname path))))
      (could-not-find))
  nil)

(defun warn-blas-without-sbcl ()
  (if (boundp 'cl-user::*cl-waffe-config*)
      (let ((config (list-to-hash (eval 'cl-user::*cl-waffe-config*))))
	(when (gethash :libblas config)
	  (warn "cl-waffe ignored :libblas option because BLAS backend is SBCL-Dependant.")))))
) ;; eval-when

;; Load libblas.dylib
(eval-when (:compile-toplevel :load-toplevel :execute)
  #+sbcl(find-and-load-libblas)
  #-sbcl(warn-blas-without-sbcl)
  )

