
(in-package :cl-waffe2/vm.generic-tensor)

;;
;; The paradigm of cl-waffe2 Dynamic Shaping.
;;

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; Dynamic Shaping is a feature which most of DL JIT Compiler supports and avoid recompiling due to the change of shape.
;; In cl-waffe2, we define Dynamic Shape as:
;;   1. OrigShape, Visible-Shape, Stride is a subject to encapsulate by LazyAxis.
;;     - These slots (a list of fixnum or LazyAxis) can contain a symbol.
;;       And Basically has these two state:
;;       Observe Mode   |  Lazy Mode
;;        (10 10 10)   <- (LazyAxis: A*B, LazyAxis: B, 10) 
;;     Once (tensor-fix-adjustable-shape tensor) was called, these LazyAxis are "fixed" and got a list of fixnum.
;;
;;   2. Adjustable Shape Declaration
;;     - we rely tensor-input-shape on determing what symbols are adjustable shape:
;;       Initially, AbstractTensor (make-input `(A B) nil) has these slots:
;;          - Shape      = (A B)
;;          - InputShape = (A B)
;;          - Stride     = (B 1)
;;          - OrigShape  = (A B)
;;
;;     After compiling the node the tensor involved, cl-waffe2 compiler collects a list of adjustable symbol given tensors.
;;     In this case: TABLE=`(A B)
;;     In order to execute the node, users have to determine the value of A and B by set-input.
;;     Provided tensor (e.g.: (10 10) AbstractTensor) cl-waffe2 compares corresponding position to InputShape
;;          - Shape      = (10 10)
;;          - InputShape = (A B)
;;     And succesfully determined as: A, B=10, 10
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(defun maybe-read-lazy-shape (lazy-shape)
  (map
   'list
   #'cl-waffe2/vm::maybe-observe-axis-no-err
   lazy-shape))

(defun observe-lazy-shape (lazy-shape)
  (map 'list #'cl-waffe2/vm:maybe-observe-axis lazy-shape))

(defclass Dynamic-Shaped-Abstract-Tensor ()
  ((orig-shape         :initform nil :initarg :shape :accessor original-shape :type list)
   (visible-shape      :initform nil :accessor tensor-visible-shape :type list)))
;; If Shape=(A*B A) -> Symbol ({|A*B| A)
(defmethod init-dynamic-shape ((self Dynamic-Shaped-Abstract-Tensor))
  (let ((result (loop for s in (original-shape self)
		      if (cl-waffe2/vm::LazyAxis-p s)
			collect (cl-waffe2/vm:lazyaxis-symbol s)
		      else
			collect s)))
    (setf (original-shape self) result)))

(defparameter *print-visible-shape* nil)

(defun shape (tensor)
  (declare (type Dynamic-Shaped-Abstract-Tensor tensor))
  (if *print-visible-shape*
      (map 'list #'(lambda (x)
		     (or (cl-waffe2/vm:symbol-lazyaxis x)
			 x))
	   (tensor-visible-shape tensor))
      (tensor-visible-shape tensor)))

(defun lazy-shape (tensor) (let ((*print-visible-shape* t)) (shape tensor)))
