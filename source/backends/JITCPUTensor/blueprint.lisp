
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

(defun cVar (tensor &key (restrict nil) (comma nil) (const nil))
  (declare (type AbstractTensor tensor))
  (cType tensor :restrict restrict :const const)
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
    (let ((view (nth axis (tensor-view tensor))))
      (if (subscript-broadcast view)
	  "0"
	  (let* ((range     (subscript-range view))
		 (direction (wf/iter:range-step range))
		 (stride    (expand-helper (nth axis (tensor-stride tensor)))))
	    (if (numberp direction)
		(format nil "~a*~a"
			(or (maybe-symbol (cl-waffe2/vm:lazyaxis-symbol direction))
			    (maybe-symbol direction))
			stride)
		stride))))))

(defun cOffset (tensor rank)
  (symb (tensor-id tensor) '_offset rank))

(defun cAref (tensor indices)
  "Reading a id of the given tensor, places a form reading an arbitary position of elements."
  (declare (type AbstractTensor tensor))
  (let ((strides (map 'list #'(lambda (axis) (cStride tensor axis)) (range 0 (dims tensor)))))
    (flet ((index-of (rank index stride)
	     (when (not (member index *solved-as-zero* :test #'string=))
	       (list
		(let* ((offset (wf/iter:range-start-index
				(wf/t:subscript-range
				 (nth rank (tensor-view tensor)))))
		       (offset (maybe-symbol (or (cl-waffe2/vm:lazyaxis-symbol offset) offset)))
		       (char
			 (format nil "(~a~a)"
				 ;; Computing offsets:
				 (if (and
				      (numberp offset)
				      (= 0 offset))
				     ""
			             (format nil "~a+" offset))
				 index)))
		  (if (and (numberp stride)
			   (= stride 1))
		      char
		      (format nil "~a*~a" char stride)))
		"+"))))
      (let ((index (apply #'concatenate 'string
			  (butlast
			   (flatten
			    (map 'list #'index-of
				 (range 0 (dims tensor)) indices strides))))))
	(format nil "~a[~a]"
		(tensor-id tensor)
		(if (string= "" index)
		    "0"
		    index))))))

(defun cAref-with-ranks (tensor indices ranks &key (name nil))
  "Reading a id of the given tensor, places a form reading an arbitary position of elements."
  (declare (type AbstractTensor tensor))
  (let ((strides (map 'list #'(lambda (axis) (cStride tensor axis)) ranks)))
    (flet ((index-of (rank index stride)
	     (when (not (member index *solved-as-zero* :test #'string=))
	       (list 
		(let* ((offset (wf/iter:range-start-index
				(wf/t:subscript-range
				 (nth rank (tensor-view tensor)))))
		       (offset (or (cl-waffe2/vm:lazyaxis-symbol offset) offset))
		       (char
			 (format nil "(~a~a)"
				 ;; Computing offsets:
				 (if (and
				      (numberp offset)
				      (= 0 offset))
				     ""
			             (format nil "~a+" (maybe-symbol offset)))
				 index)))
		  (if (and (numberp stride)
			   (= stride 1))
		      char
		      (format nil "~a*~a" char stride)))
		"+"))))
      (let ((index (apply #'concatenate 'string
			  (butlast
			   (flatten
			    (map 'list #'index-of ranks indices strides))))))
	(format nil "~a[~a]"
		(or name (tensor-id tensor))
		(if (string= index "")
		    "0"
		    index))))))


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

(defun cFunction (function-name adjustable-shape arguments &key (displace-to-list nil))
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
		  do (cVar arg :restrict (= 1 (count (tensor-id arg) arguments :key #'tensor-id))
			       :comma (not (= nth (1- (length arguments))))
			       :const (and (not (null displace-to-list))
					   (not (member (tensor-id arg) displace-to-list)))))
	    (write-buff ")"))))
    (format nil "void ~a~a" function-name arguments-form)))

