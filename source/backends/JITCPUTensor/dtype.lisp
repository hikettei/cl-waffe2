
(in-package :cl-waffe2/backends.jit.cpu)

;; cl-waffe2 dtype -> C Dtype

;; single-flaot -> float
;; double-float -> double
;;        ...


;;
;; (c++defun aaa ((c++type (dtype tensor)) ... ...)
;;    (inst-set ())
;;    (inst-load ()))
;;

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

(defun cType (tensor &key (restrict nil))
  (declare (type AbstractTensor tensor)
	   (type boolean restrict))
  (typecase tensor
    (JITCPUTensor
     (write-buff "~a~a"
		 (dtype->ctype (dtype tensor))
		 (if restrict
		     " restrict * "
		     "* ")))
    (JITScalarTensor
     (write-buff "~a "		
		 (dtype->ctype (dtype tensor))))
    (T
     (error "cType: the given tensor isn't JITAble.
~a" tensor))))

(defun cVar (tensor &key (restrict nil) (comma nil))
  (declare (type AbstractTensor tensor))
  (cType tensor :restrict restrict)
  (write-buff "~a~a" (tensor-id tensor)
	      (if comma ", " "")))

(defun cStride (tensor &key (comma nil))
  (declare (type JITCPUTensor))
  (write-buff "int32_t restrict * ~a_STRIDE~a" (tensor-id tensor)
	      (if comma ", " "")))

