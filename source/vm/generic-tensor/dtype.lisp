
(in-package :cl-waffe2/vm.generic-tensor)

(deftype dtype-t ()
  "Supported dtype is following:
:double
:float
:uint32
:uint16
:uint8
:int32
:int16
:int8
:bit"
  `(and keyword
	;; More will be added...
	(member :double
		:float
		:uint64
		:int64
		:uint32
		:uint16
		:uint8
		:int32
		:int16
		:int8
		:bit)))

(declaim (ftype (function (dtype-t) t) dtype->lisp-type))
(defun dtype->lisp-type (dtype)
  (declare (type dtype-t dtype))
  (case dtype
    (:double 'double-float)
    (:float  'single-float)
    (:uint64 '(unsigned-byte 64))
    (:int64  '(signed-byte 64))
    (:uint32 '(unsigned-byte 32))
    (:uint16 '(unsigned-byte 16))
    (:uint8  '(unsigned-byte 8)) 
    (:int32  '(signed-byte 32))
    (:int16  '(signed-byte 16))
    (:int8   '(signed-byte 8))
    (:bit    'bit)))

(defun dtype-of (scalar)
  (typecase scalar
    (single-float :float)
    (double-float :double)
    ;; FIXME
    (fixnum :int32)
    (integer :int64)
    (T (error "Cannot infer the type of ~a" scalar))))

(defun coerce-into-dtype (scalar)
  (coerce scalar (dtype-of scalar)))

(defstruct Lazy-Variable
  (variable)
  (dtype))

(defmethod print-object ((model lazy-variable) stream)
  (format stream "<LazyVariable> (the ~(~a~) ~a)"
	  (lazy-variable-dtype model)
	  (lazy-variable-variable model)))

(defun read-lazy-var (lazy-var)
  (declare (type lazy-variable lazy-var))

  (let ((result (read-symbol (lazy-variable-variable lazy-var))))
    (if (numberp result)
	(coerce result (lazy-variable-dtype lazy-var))
	result)))

(defun coerce-lazy (scalar dtype)
  "If scalar is number, coerce to dtype, otherwise lazily evalute"
  (if (numberp scalar)
      (coerce scalar dtype)
      (if (symbolp scalar)
	  (make-lazy-variable :variable scalar :dtype dtype)
	  (cl-waffe2/vm:make-lazyaxis scalar))))


