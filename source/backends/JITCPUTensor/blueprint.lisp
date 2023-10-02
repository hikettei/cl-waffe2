
(in-package :cl-waffe2/backends.jit.cpu)

(defclass CPUJIT-Blueprint ()
  ((opecode :initform nil :type symbol :accessor blueprint-opecode)
   (use-vars :initform nil :type list :accessor  blueprint-use-var))
  (:documentation "
## [class] CPUJIT-Blueprint
Stores information related to JIT Compiling.
"))

(defgeneric translate-op (opcode opast &rest args) (:documentation "
## [generic] translate-op
A method returning corresponding instruction given opcode
"))


(defstruct (Instruction
	    (:constructor make-inst (inst-type function-name displace-to function-arguments)))
  "
 displace-to
   A[...]     = function-name(function-arguments)
 "
  (type inst-type :type (and keyword (member :modify :apply :set :ignore)))
  (fname function-name :type string)
  (displace-to displace-to :type AbstractTensor)
  (args function-arguments :type list))


(defun cVar (tensor &key (restrict nil) (comma nil) (pointer nil))
  (declare (type AbstractTensor tensor))
  (cType tensor :restrict restrict :pointer pointer)
  (write-buff "~a~a" (tensor-id tensor)
	      (if comma ", " "")))

(defun cStride (tensor axis)
  (declare (type JITCPUTensor tensor)
	   (type fixnum axis))
  (labels ((expand-helper (stride)
	     (trivia:match stride
	       ((type fixnum)
		stride)		 
	       ((list 'the 'fixnum form)
		(expand-helper form))
	       ((list '* a b)
		(format nil "~a*~a"
			(expand-helper a)
			(expand-helper b)))
	       ((list _ b)
		(let ((res (format nil "~a" b)))
		  (subseq res 1 (length res))))
	       (T
		(error "cStride: Encountered Unknown Stride Syntax ~a" stride)))))
    (expand-helper (nth axis (tensor-stride tensor)))))

(defun cPointer (tensor)
  (symb (tensor-id tensor) '-ptr))



