
(in-package :cl-waffe2/vm.generic-tensor)

;; Here we provide two macros:
;; define-tensor (Tensors and Backend are strongly combined.)
;; CFFI-Style
;; Column-Major And Row-Major
;; TODO: Detect it: (make-tensor `(10 a)) <- use (make-input)

(defparameter *using-backend*
  `()
  "cl-waffe searches for computation nodes in the following order and uses the first one it finds. (Priority1 Priority2 ...)
Default: `(cl-waffe2/vm.generic-tensor:CPUTensor)
PriorityN must be a subclass of cl-waffe2/vm.generic-tensor:AbstractTensor")

(defun order-p (name)
  (declare (type keyword name))
  (or (eql name :column) (eql name :row)))

;; cl-xmatrixからView関連のコードを移植しておく

(defclass AbstractTensor ()
  ((nodes :initarg :nodes :initform nil :reader tensor-nodes :type list)

   ;; MultiDimensional
   (orig-shape :initarg :shape :initform nil :reader original-shape :type list)
   (stride :initform nil :accessor tensor-stride :type list)
   (visible-shape :initform nil :reader shape :accessor tensor-visible-shape :type list)
   (view :initarg :view :initform nil :accessor tensor-view :type list)
   
   (projected-p :initarg :projected-p :initform nil :type boolean :reader tensor-projected-p)
   (scalar-p :initarg :scalar-p :initform nil)
   
   (vec :initarg :vec :initform nil :accessor tensor-vec)
   (dtype :initform :float :initarg :dtype :reader dtype)

   ;; Building Computation Nodes
   (backward  :initform nil :accessor tensor-backward)
   (state     :initform nil :accessor tensor-state)
   (variables :initform nil :accessor tensor-variables)

   (tensor-id :initform (gensym "TID") :accessor tensor-id)
   (nth-value :initform 0 :accessor tensor-out-n :type fixnum)
   
   (requires-grad :initform nil :initarg :requires-grad :type boolean)
   (order :initarg :order :initform :column :type (satisfies order-p) :accessor order)
   (trace-state :initform nil) ;; For Optimizing Computation Node

   (facet :initarg :facet :initform :exist :type (member :exist :input) :accessor tensor-facet)
   (named :initarg :named :type keyword :accessor tensor-name)

   ))

;; (defmacro variable  ())
;; (defmacro parameter ())

(defmethod initialize-instance :before ((tensor AbstractTensor) &rest initargs &key &allow-other-keys)
  (let ((scalar-p   (getf initargs :scalar-p))
	(view       (getf initargs :view))
	(order      (getf initargs :order))
	(orig-shape (getf initargs :shape)))
    
    (setf (slot-value tensor 'orig-shape) orig-shape)
    (setf (slot-value tensor 'projected-p) (getf initargs :projected-p))
    
    (cond
      ((eql (getf initargs :facet) :input)
       (setf (tensor-view tensor)
	     (parse-view-subscripts tensor (getf initargs :past-view) (or view `(t))))
       (setf (tensor-visible-shape tensor)
	     (compute-visible-shape orig-shape (tensor-view tensor)))
       nil)
      (T
       (when (not scalar-p)
	 (setf (tensor-stride tensor) (calc-strides orig-shape order))
	 ;; parse-view-subscripts <- safety 0...
	 (setf (tensor-view tensor) (parse-view-subscripts tensor (getf initargs :past-view) (or view `(t))))
	 (setf (tensor-visible-shape tensor)
	       (compute-visible-shape orig-shape (tensor-view tensor)))
	 nil)))))

(defmacro assure-dimensions (mat1 mat2)
  "Does nothing if mat1 and mat2 has a completely the same shape, otherwise throws shaping-error."
  `(if (equal (the list (shape ,mat1)) (the list (shape ,mat2)))
       t
       (shaping-error "Assertion Failed because two matrices ~a and ~a couldn't operated together." (shape ,mat1) (shape ,mat2))))

(defmethod calc-strides ((shape list) (order (eql :column)))
  "Computes column-major-strides"
  (column-major-calc-strides shape))

(defmethod calc-strides ((shape list) (order (eql :row)))
  "Computes row-major-strides"
  (row-major-calc-strides shape))

(defun make-tensor (shape-or-scalar
		    &key
		      (requires-grad nil)
		      (dtype :float)
		      (vec  nil)
		      (view nil)
		      (order :column))
  "The function make-tensor creates a new tensor of first-priority of *using-backend*"
  (declare (type list view)
	   (ignore vec)) ;; vec is working. at initialize-instance :before
  (if (typep shape-or-scalar 'list)
      (make-instance (car *using-backend*)
		     :dtype dtype
		     :order order
		     :requires-grad requires-grad
		     :shape shape-or-scalar
		     :projected-p nil
		     :view view)
      (make-instance 'ScalarTensor
		     :scalar-p t
		     :vec shape-or-scalar
		     :shape nil
		     :dtype dtype
		     :requires-grad requires-grad
		     :projected-p nil
		     :view view)))

;; It is allowed: (make-input `(batch-size 512))
(defun make-input (shape-or-scalar
		   named
		   &key
		     (dtype :float)
		     (order :column))
  "named ... variable-name (keyword)"
  (if (typep shape-or-scalar 'list)
      (make-instance (car *using-backend*)
		     :dtype dtype
		     :order order
		     :shape shape-or-scalar
		     :named named
		     :facet :input)
      (make-instance 'ScalarTensor
		     :scapar-p t
		     :vec shape-or-scalar
		     :shape nil
		     :dtype dtype
		     :projected-p nil
		     :named named
		     :facet :input)))

  
(defun view (tensor &rest subscripts)

  ;; TODO: When tensor is scalar, return error.

  (make-instance (car *using-backend*)
		 :dtype (dtype tensor)
		 :order (order tensor)
		 :requires-grad (slot-value tensor 'requires-grad)
		 :shape (slot-value tensor 'orig-shape)
		 :projected-p t
		 :past-view (tensor-view tensor)
		 :view subscripts
		 :facet (tensor-facet tensor)
		 :vec (slot-value tensor 'vec)))

(defmethod print-object ((tensor AbstractTensor) stream)
  (format stream
	  "{~a[~(~a~)] ~a ~a
  ~a <- Display Contents
  :state :forward :requires-grad T :backward <AddNode>}"
	  (class-name (class-of tensor))
	  (dtype tensor)
	  (if (slot-value tensor 'scalar-p)
	      ""
	      (cond
		((null (slot-value tensor 'projected-p))
		 ;; Orig-Shape
		 (format nil ":shape ~a" (shape tensor)))
		(T
		 ;; It has a view
		 (format nil ":shape ~a -> :view ~a -> :visible-shape ~a"
			 (slot-value tensor 'orig-shape)
			 (slot-value tensor 'view)
			 (shape tensor)))))
	  (if (eql (tensor-facet tensor) :input)
	      (format nil ":named ~a" (tensor-name tensor))
	      "")
	  "(TODO)"
	  ))

