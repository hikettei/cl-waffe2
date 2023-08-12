
(in-package :cl-waffe2/vm.generic-tensor)

;; cache.lisp is used to reuse the result of **call-with-view** in compling time, which consists larger part of generated code, and most reusable part.

;; =======================================================================
;; Goal: `3 Layers MLP's compiling time of forward and backward` << `0.5sec`.

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


(defparameter *compiled-function-cache* (make-hash-table))
(defparameter *compiled-jit-function-cache* (make-hash-table))

(defun reset-compiled-function-cache! ()
  "
## [function] reset-compiled-function-cache!
"
  (setf *compiled-function-cache* (make-hash-table))
  (setf *compiled-jit-function-cache* (make-hash-table))
  t)

;; Problem: Compiling !softmax called with other nodes is slightly slow.

(defvar *kernel-storeroom* nil "An storeroom to record all kernels used in compiling time.") ;; Corresponds to: (labels ((SinNode-... )) ... )

;; An unit of compiled funciton.
(defstruct Compiled-Kernel
  (name nil :type symbol)            ;; SinNode-CPUTENSOR
  (body nil :type list)              ;; (named-lambda ... () ...)
  (cache-when-compiled nil :type boolean)
  (cache-p nil :type boolean)
  (call-with-view nil :type (or null Ranked-Loop))
  (args nil :type list)
  (view-route nil :type list)
  (self nil)) ;; 2D 3D Flatten ...

;; the maximum length of symbol-name used in CL shoule be 512? i dont remember ...
;; Memo:
(defun make-kernel-name (tensor reductable-p-map)
  "The naming rule of kernel functions is following:

[NodeName] - [Backend-Name] - [TensorViewName1] - [TensorViewName2] ...

where TensorViewNameN is:

[4D -> 3D -> 2D -> ... -> end-name]

end-name = [Flatten] or 1D

For Example:

(!sin (randn `(100 100))) -> `SinNode-LispTensor-Flatten`

(!sin (!view (randn `(100 100) `(1 -1) t)))
-> `SinNode-LispTensor-2D->Flatten`

TensorViewNameN depicts the path call-with-view traced.
"
  (flet ((template (&optional dim)
	   (if dim
	       (symb ;; {->(DIM)-BackendName[Dype]}
		'{
		(intern (format nil "->(~a)-" dim))
		(class-name (class-of tensor))
		'[
		(intern (symbol-name (dtype tensor)))
		']
		'})
	       (symb ;; {BackendName[Dtype]}
		'{
		(class-name (class-of tensor))
		'[
		(intern (symbol-name (dtype tensor)))
		']
		'}))))

    (when (scalar-p tensor)
      (return-from make-kernel-name (template)))

    (let ((name-list))
      (loop named trace-loop
	    for dim downfrom (1- (length (shape tensor))) to 0
	    if (nth dim reductable-p-map)
	      do (push (template 'flatten) name-list)
		 (return-from trace-loop)
	    else
	      do (push (template dim) name-list))
      ;; call-with-view traces following
      ;; n-1 n-2 ... 2 1 0th dim.
      ;; Caching Based on Tensor Strides (Because it is inlined)
      (apply #'symb (reverse name-list) (list (intern (format nil "~a~a" (actual-shape tensor) (tensor-permute-order tensor))))))))

(defun kernel-name (compiled-kernel)
  (symb
   (compiled-kernel-name compiled-kernel)
   (if (movetensor-p (compiled-kernel-self compiled-kernel))
       (cl-waffe2/base-impl:movetensor-ignore-me (compiled-kernel-self compiled-kernel))
       '-)
   (apply
    #'symb
    (map 'list #'(lambda (x) (make-kernel-name x (compiled-kernel-view-route compiled-kernel))) (compiled-kernel-args compiled-kernel)))))

(defun cache-kernel-form (compiled-function)
  (let ((fbody (cdar (compiled-kernel-body compiled-function))))
    `(,(kernel-name compiled-function)
      ,@(tensor->id (cdr fbody) (compiled-kernel-args compiled-function)))))

(defparameter *print-compiled-function* nil)

(defun make-funcallable-kernel (compiled-function compile-option)
  (declare (type Compiled-Kernel compiled-function))
  (let ((fbody (cdar (compiled-kernel-body compiled-function))))
    (compile nil
	     (let ((out (tensor->id (cdr fbody) (compiled-kernel-args compiled-function))))
	       (let ((args (car out))
		     (decl (second out))
		     (body (cddr out)))
		 (let ((out `(lambda ,args ,decl (declare ,compile-option) ,@body)))
		   (if *print-compiled-function*
		       (print out)
		       out)))))))

(defun place-cached-kernels (&rest body)
  "
Reading *kernel-storeroom*, the function expands the form below.
(progn
;; [LUT PLACE]
(defun SinNode-CPUTensor-3D ()
    ...)

(defun SinNode-CPUTensor-2D ()
    ...)

...

;; Forward/Backward Process continues...
)"
  ;; Expand *kernel-storeroom*

  (let ((caches (make-hash-table)))
    (dolist (fn *kernel-storeroom*)
      (setf (gethash (kernel-name fn) caches) (cache-kernel-form fn)))
    `(labels (,@(loop for body being the hash-values in caches
		      collect body))
       ;; TODO: Prune this:
       ,@(loop for name being the hash-keys in caches
	       collect `#',name)
       ,@body)))

(defun tensor->id (body args)
  (map-tree
   #'(lambda (obj)
       (typecase obj
	 (AbstractTensor
	  (if (find (tensor-id obj) args :key #'tensor-id)
	      (tensor-id obj)
	      obj))
	 (T
	  obj)))
   body))

(defmacro funcall-kernel (kernel-function &rest inputs)
  "A replacement of (funcall fw-compiled)"
  `(funcall ,@(tensor->id (compiled-kernel-body kernel-function) (compiled-kernel-args kernel-function)) ,@inputs))

(defmacro call-cache-fn (kernel-function &rest inputs)
  `(,(kernel-name kernel-function) ,@inputs))

(defmacro call-kernel (kernel-function &rest inputs)
  (if (compiled-kernel-cache-p kernel-function)
      `(call-cache-fn ,kernel-function ,@inputs)
      `(funcall-kernel ,kernel-function ,@inputs)))


;; For toplevel
(defun funcall-cached-function (kernel-function compile-option &rest args)
  (declare (type Compiled-Kernel kernel-function))

  (let* ((function-name (compiled-kernel-name kernel-function))
	 ;; MoveTensor is a special node that cl-waffe2 will dynamically ignore/use depending on the situation.
	 ;; So, we dont wanna use :cache-when-compiled option.
	 (movep (movetensor-p (compiled-kernel-self kernel-function)))
	 (function-name (if movep
			    (symb function-name '- (cl-waffe2/base-impl:movetensor-ignore-me (compiled-kernel-self kernel-function)))
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
	 ;; MoveTensor is a special node that cl-waffe2 will dynamically ignore/use depending on the situation.
	 ;; So, we dont wanna use :cache-when-compiled option.
	 (movep (movetensor-p (compiled-kernel-self kernel-function)))
	 (function-name (if movep
			    (symb function-name '- (cl-waffe2/base-impl:movetensor-ignore-me (compiled-kernel-self kernel-function)))
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

