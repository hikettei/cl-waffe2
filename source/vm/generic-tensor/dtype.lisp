
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
    (:uint32 '(unsigned-byte 32))
    (:uint16 '(unsigned-byte 16))
    (:uint8  '(unsigned-byte 8)) 
    (:int32  '(signed-byte 32))
    (:int16  '(signed-byte 16))
    (:int8   '(signed-byte 8))
    (:bit    'bit)))

