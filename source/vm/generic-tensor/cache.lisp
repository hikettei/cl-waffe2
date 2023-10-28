
(in-package :cl-waffe2/vm.generic-tensor)

;;
;; ~~ CALL_WITH_VIEW => FUNCTION ~~
;;

;; cache.lisp is used to reuse the result of **call-with-view** in compling time, which consists larger part of generated code, and most reusable part.

;; =======================================================================

;; Compiling time with cache.lisp is approximated as:
;; ```
;; O((the number of kernel types used in nodes))
;; ```
;; while without cache.lisp:
;; ```
;; O((the number of operation))
;; ```
;; =======================================================================

;; the number of kernel types used in nodes <-> compiling time of kernels in *kernel-storeroom*

;; **This Parameter isn't used anymore?**
;; (defparameter *cache-directory* "~/.cache/cl-waffe2/")

;; The file cache.lisp provides an optimized kernel compiler for acceptor.lisp

;;
;; (defnode (SinNode ...) ) is named as [SinNode-CPUTensor-2D]
;; being defined as like:
;; (defun Sinnode-CpuTensor-2D (...)
;;    (optimize ...)
;;    (lambda (..) ...))
;; We store such like functions to reduce compile-time.
;;

;; 1. Create a LUT to store generated and reusable s-expression.
;; ==================================================================
;; A template form of compiled cl-waffe2 programs:
;; (labels ((SinNode-CPUTensor-3D () ;; Storeroom of compiled kernels used
;;             ...
;;           )
;;          (SinNode-CPUTensor-2D ()
;;             ...
;;           ))
;;
;; (forward/backward networks continues) <- Place SinNode-CPUTensor-3D for example.
;; )
;; ==================================================================

;; What I want to do?: the compiling time of (!sin (!sin (!sin x))) and (!sin x) should be approximately the same, because they're working on the same code ignoring indicating pointer.

;; TODO: build/proceed must use the same algorithm to cache compiled functions


(defparameter *compiled-function-cache* (make-hash-table :test #'equal))
(defparameter *compiled-jit-function-cache* (make-hash-table))

(defun reset-compiled-function-cache! ()
  "
## [function] reset-compiled-function-cache!
"
  (setf *compiled-function-cache* (make-hash-table :test #'equal))
  (setf *compiled-jit-function-cache* (make-hash-table))
  t)

(defstruct Compiled-Kernel
  "All forward method must return this structure. Compiled-Kernel is an blueprint (or stores compiled functions) to generate cl-waffe2 IR"
  (op   nil :type (or null function))
  (name nil :type symbol)            ;; SinNode-CPUTENSOR
  (body nil :type list)              ;; (named-lambda ... () ...)
  (cache-when-compiled nil :type boolean)
  (cache-p nil :type boolean)
  (call-with-view nil :type (or null))
  (args nil :type list)
  (view-route nil :type list)
  (self nil)
  (cache-additional-id nil :type list)) ;; 2D 3D Flatten ...

(defun make-funcallable-kernel (compiled-function compile-option)
  (declare (type Compiled-Kernel compiled-function))
  ;; If the Compiled-kernel already provides any op as a lambda function, -> use this
  ;; Otherwise -> compile and cache
  (or (compiled-kernel-op compiled-function)
      (compile
       nil
       ;; [TODO]
       ;; Expected body: (lambda args body)
       ;;                            ^ Insert Compile-Option
       (let ((body (compiled-kernel-body compiled-function)))
	 `(,(car body)
	   ,(second body)
	   (declare ,compile-option)
	   ,@(cddr body))))))

(defun funcall-cached-function (kernel-function compile-option &rest args)
  (declare (type Compiled-Kernel kernel-function))

  (let* ((function-name (compiled-kernel-name kernel-function))
	 (more-ids (compiled-kernel-cache-additional-id kernel-function))
	 (function-name (if more-ids
			    `(,function-name ,more-ids)
			    function-name))
	 (target (lut-search-function
		  *compiled-function-cache*
		  function-name
		  args)))
    (if (null target)
	(let ((compiled-function (make-funcallable-kernel kernel-function compile-option)))
	  (when (compiled-kernel-cache-when-compiled kernel-function)
	    (lut-search-function
	     *compiled-function-cache*
	     function-name
	     args
	     :setme compiled-function))
	  (apply compiled-function args))
	(apply target args))))

(defun find-cached-function (kernel-function compile-option &rest args)
  (declare (type Compiled-Kernel kernel-function))

  (let* ((function-name (compiled-kernel-name kernel-function))
	 (more-ids (compiled-kernel-cache-additional-id kernel-function))
	 (function-name (if more-ids
			    `(,function-name ,more-ids)
			    function-name))
	 (target (lut-search-function
		  *compiled-function-cache*
		  function-name
		  args)))
    (if (null target)
	(let ((compiled-function (make-funcallable-kernel kernel-function compile-option)))
	  (when (compiled-kernel-cache-when-compiled kernel-function)
	    (lut-search-function
	     *compiled-function-cache*
	     function-name
	     args
	     :setme compiled-function))
	  compiled-function)
	target)))

