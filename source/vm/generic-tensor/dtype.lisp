
(in-package :cl-waffe2/vm.generic-tensor)

(deftype dtype-t ()
  `(and keyword
	;; More will be added...
	(member :double
		:float
		:uint32
		:int32
		:bit)))

(declaim (ftype (function (dtype-t) t) dtype->lisp-type))
(defun dtype->lisp-type (dtype)
  (declare (type dtype-t dtype))
  (case dtype
    (:double 'double-float)
    (:float  'single-float)
    (:uint32 '(unsigned-byte 32))
    (:int32  '(singed-byte 32))
    (:bit    'bit)))

