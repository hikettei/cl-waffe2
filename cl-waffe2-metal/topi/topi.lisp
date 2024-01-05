
(in-package :cl-waffe2/backends.metal)

(defparameter *all-dtype-case* `(float int64-t int32-t int16-t int8-t uint64-t uint32-t uint16-t uint8-t))

(defun dtype->mtype (dtype)
  (declare (type keyword dtype))
  (case dtype
    (:float 'float)
    (:int64 'int64-t)
    (:int32 'int32-t)
    (:int16 'int16-t)
    (:int8  'int8-t)
    (:uint64 'uint64-t)
    (:uint32 'uint32-t)
    (:uint16 'uint16-t)
    (:uint8  'uint8-t)
    (T (error "dtype->mtype: not supported dtype: ~a" dtype))))

(defmacro def-metal-caller (name (&rest tensors) (&rest args))
  `(defun ,name (dtype n ,@args)
     (with-tensor-ptrs (,@(loop for tensor in tensors
				collect
				`(,tensor ,tensor)))
       (case (dtype->mtype dtype)
	 ,@(loop for dtype in *all-dtype-case*
		 collect
		 `(,dtype
		   (%funcall-metal
		    (get-kernel ',(symb name '- dtype))
		    :args (list ,@args)
		    :kcount n)))))))

(defun mfuncall (function &rest args)
  (funcall function (map 'list #'tensor-vec args)))

(define-compiler-macro mfuncall (function &rest args)
  `(funcall ,function ,@(loop for x in args
			      collect
			      `(tensor-vec ,x))))

