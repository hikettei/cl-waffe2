
(in-package :cl-waffe2/vm.generic-tensor)

;; Here we provide two macros:
;; define-tensor (Tensors and Backend are strongly combined.)
;; CFFI-Style
;; Column-Major And Row-Major

(defparameter *using-backend*
  `(cl-waffe2/vm.generic-tensor:CPUTensor)
  "cl-waffe searches for computation nodes in the following order and uses the first one it finds. (Priority1 Priority2 ...)
Default: `(cl-waffe2/vm.generic-tensor:CPUTensor)
PriorityN must be a subclass of cl-waffe2/vm.generic-tensor:AbstractTensor")

(defun order-p (name)
  (declare (type keyword name))
  (or (eql name :column) (eql name :row)))

;; cl-xmatrixからView関連のコードを移植しておく

(defclass AbstractTensor ()
  ((nodes :initarg :nodes :initform nil :reader tensor-nodes :type list)
   (orig-shape :initarg :shape :initform nil :reader original-shape :type list)
   (stride :initform nil :reader tensor-stride :type list)
   (visible-shape :initform nil :reader shape :type list)
   (view :initarg :view :initform nil :reader tensor-view :type list)
   (projected-p :initarg :projected-p :initform nil :type boolean :reader tensor-projected-p)
   (scalar-p :initarg :scalar-p :initform nil)
   (vec :initarg :vec :initform nil)
   (dtype :initform :float :initarg dtype :reader dtype)

   ;; Building Computation Nodes
   (previous-state :initform nil :accessor tensor-prev-state)
   (previous-form  :initform nil :accessor tensor-prev-form)
   (variables      :initform nil :accessor tensor-variables)
   
   (requires-grad :initform nil :initarg :requires-grad :type boolean)
   (order :initarg :order :initform :column :type (satisfies order-p) :accessor order)
   (trace-state :initform nil) ;; For Optimizing Computation Node

   ))

(defun compute-visible-shape (orig-shape view)
  (loop for o in orig-shape
	for v in view
	collect (- (view-endindex   v o)
		   (view-startindex v 0))))

(defmethod initialize-instance :after ((tensor AbstractTensor) &rest initargs &key &allow-other-keys)
  (declare (ignore initargs))
  (with-slots ((stride stride) (order order) (visible-shape visible-shape) (view view)) tensor
    ;; visible area
    (setf stride (calc-strides tensor order))
    ;; parse-view-subscripts <- safety 0...
    (setf view   (parse-view-subscripts tensor (or view `(t))))
    (setf visible-shape (compute-visible-shape (original-shape tensor) view))
    nil))

(defmacro assure-dimensions (mat1 mat2)
  "Does nothing if mat1 and mat2 has a completely the same shape, otherwise throws shaping-error."
  `(if (equal (the list (shape ,mat1)) (the list (shape ,mat2)))
       t
       (shaping-error "Assertion Failed because two matrices ~a and ~a couldn't operated together." (shape ,mat1) (shape ,mat2))))

(defmethod calc-strides ((tensor AbstractTensor) (order (eql :column)))
  "Computes column-major-strides"
  (column-major-calc-strides (slot-value tensor 'orig-shape)))

(defmethod calc-strides ((tensor AbstractTensor) (order (eql :row)))
  "Computes row-major-strides"
  (row-major-calc-strides (slot-value tensor 'orig-shape)))

;; Tensors must support displace-to

;; (defmethod print-object ((tensor AbstractTensor) stream))


(defun make-tensor (shape-or-scalar
		    &key
		      (requires-grad nil)
		      (dtype :float)
		      (view `(t))
		      (order :column))
  "The function make-tensor creates a new tensor of first-priority of *using-backend*"
  (declare (type list view))
  (let ((shape (if (typep shape-or-scalar 'list)
		   shape-or-scalar
		   `(1))))
    (make-instance (car *using-backend*)
		   :dtype dtype
		   :order order
		   :requires-grad requires-grad
		   :shape shape
		   :projected-p nil
		   :view view)))

(defun view (tensor &rest subscripts)

  )
