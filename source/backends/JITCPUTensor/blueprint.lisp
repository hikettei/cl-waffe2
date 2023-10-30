
(in-package :cl-waffe2/backends.jit.cpu)

(defclass CPUJIT-Blueprint ()
  ((foreign-kernel)
   (opecode :initform nil :type symbol :accessor blueprint-opecode)
   (use-vars :initform nil :type list :accessor  blueprint-use-var))
  (:documentation "
## [class] CPUJIT-Blueprint
Stores information related to JIT Compiling.
"))

(defstruct (Instruction
	    (:constructor make-inst (inst-type function-name displace-to function-arguments)))
  "
 displace-to
   A[...]     = function-name(function-arguments)

type:
 - modify  A fname B
 - apply   A = fname(B)
"
  (type inst-type :type (and keyword (member :modify :apply)))
  (fname function-name :type string)
  (displace-to displace-to :type AbstractTensor)
  (args function-arguments :type list))

(defgeneric load-instructions (node &rest inputs))

(defun collect-variables (instructions)
  (let ((result))
    (dolist (inst instructions)
      (dolist (tensor `(,@(instruction-args inst) ,(instruction-displace-to inst)))
	(when (not (find (tensor-id tensor) result :key #'tensor-id))
	  (push tensor result))))
    result))

(defun cPointer (tensor)
  (symb (tensor-id tensor) '-ptr))

(defun cVar (tensor &key (restrict nil) (comma nil))
  (declare (type AbstractTensor tensor))
  (cType tensor :restrict restrict)
  (write-buff "~a~a"
	      (tensor-id tensor)
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
		  (c-name (subseq res 1 (length res)))))
	       (T
		(error "cStride: Encountered Unknown Stride Syntax ~a" stride)))))
    (let ((view (force-list (nth axis (tensor-view tensor)))))
      (if (and (listp view)
	       (eql (car view) :broadcast))
	  "0"
	  (expand-helper (nth axis (tensor-stride tensor)))))))

(defun cOffset (tensor rank)
  (symb (tensor-id tensor) '_offset rank))

(defun cAref (tensor indices)
  "Reading a id of the given tensor, places a form reading an arbitary position of elements."
  (declare (type AbstractTensor tensor))
  (let ((strides (map 'list #'(lambda (axis) (cStride tensor axis)) (range 0 (dims tensor)))))
    (flet ((index-of (rank index stride)
	     (list (format nil "(~a~a)*~a"
			   (if (= 0 (cl-waffe2/vm.generic-tensor::compute-visible-start-idx
				     (force-list (nth rank (tensor-view tensor)))))
			       ""
			       (format nil "~a+" (cOffset tensor rank)))
			   index stride)
		   "+")))
      (format nil "~a[~a]"
	      (tensor-id tensor)
	      (apply #'concatenate 'string
		     (butlast
		      (flatten
		       (map 'list #'index-of
			    (range 0 (dims tensor)) indices strides))))))))

(defun cAref-with-ranks (tensor indices ranks)
  "Reading a id of the given tensor, places a form reading an arbitary position of elements."
  (declare (type AbstractTensor tensor))
  (let ((strides (map 'list #'(lambda (axis) (cStride tensor axis)) ranks)))
    (flet ((index-of (rank index stride)
	     (list (format nil "(~a~a)*~a"
			   (if (= 0 (cl-waffe2/vm.generic-tensor::compute-visible-start-idx
				     (force-list (nth rank (tensor-view tensor)))))
			       ""
			       (format nil "~a+" (cOffset tensor rank)))
			   index stride)
		   "+")))
      (format nil "~a[~a]"
	      (tensor-id tensor)
	      (apply #'concatenate 'string
		     (butlast
		      (flatten
		       (map 'list #'index-of ranks indices strides))))))))


(defun solve-depends-on (tensor)
  "Returns a list of indices tensor depends on"
  (declare (type AbstractTensor tensor))
  (let ((strides (map 'list #'(lambda (axis) (cStride tensor axis)) (range 0 (dims tensor)))))
    (flet ((index-of (rank stride)
	     (if (or (and (stringp stride)
			  (string= stride "0"))
		     (and (numberp stride)
			  (= stride 0)))
		 nil
		 rank)))
      (loop for s in (map 'list #'index-of (range 0 (dims tensor)) strides)
	    if s collect s))))

(defun cFunction (function-name adjustable-shape arguments)
  "Header:
void function-name (int size, float * restrict x1, int stride, int offset, float* x2 ...)

  Returns the definition form of given function."
  (let ((arguments-form
	  (with-compiling-mode
	    (write-buff "(")
	    (dolist (shape adjustable-shape)
	      (write-buff "uint32_t ~a, " (c-name (format nil "~a" shape))))
	    (loop for arg in arguments
		  for nth upfrom 0
		  ;; TENSOR_PTR OFFSET3 OFFSET2 OFFSET1
		  do (cVar arg :restrict (= 1 (count (tensor-id arg) arguments :key #'tensor-id)) :comma t)
		     (dotimes (rank (dims arg))
		       (write-buff "const uint32_t ~a~a"
				   (cOffset arg rank)
				   ;; Judge if the last or not
				   (if (and (= rank (1- (dims arg)))
					    (= nth  (1- (length arguments))))
				       ""
				       ", "))))
	    (write-buff ")"))))
    (format nil "void ~a~a" function-name arguments-form)))

