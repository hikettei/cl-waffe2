
(in-package :cl-waffe2/backends.jit.cpu)

;; LUT: cl-waffe2 dtype -> C type

;; Memo: https://zenn.dev/mod_poppo/articles/vectorization-and-restrict
(defun dtype->ctype (dtype)
  (declare (type keyword dtype))
  (case dtype
    (:double "double")
    (:float "float")
    (:int32 "int32_t")
    (:int16 "int16_t")
    (:int8 "int8_t")
    (:uint32 "uint32_t")
    (:uint16 "uint16_t")
    (:uint8 "uint8_t")
    (:bit "bool")
    (T (error "dtype->ctype: Attempted to generate C codes
but failed because cl-waffe2 encountered an unsupported dtype: ~a" dtype))))

(defun cType (tensor &key (restrict nil) (const nil))
  (declare (type AbstractTensor tensor)
	   (type boolean restrict))
  (typecase tensor
    (JITCPUTensor
     (write-buff "~a~a~a"
		 (if const "const " "")
		 (dtype->ctype (dtype tensor))
		 (if restrict
		     "* restrict "
		     "* ")))
    (T
     (error "cType: the given tensor isn't JITAble.
~a" tensor))))

