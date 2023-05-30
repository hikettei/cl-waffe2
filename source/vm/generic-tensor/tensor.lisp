
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
   
   (vec :initarg :vec :initform nil :reader vec :writer write-vec)
   (dtype :initform :float :initarg :dtype :reader dtype)

   ;; Building Computation Nodes
   (backward  :initform nil :accessor tensor-backward)
   (state     :initform nil :accessor tensor-state)
   (variables :initform nil :accessor tensor-variables)

   (tensor-id :initform (gensym "TID") :accessor tensor-id)
   (nth-value :initform 0 :accessor tensor-out-n :type fixnum)

   (grad :initform nil :reader grad :writer set-grad)
   (requires-grad :initform nil :initarg :requires-grad :type boolean)
   (order :initarg :order :initform :column :type (satisfies order-p) :accessor order)
   (trace-state :initform nil) ;; For Optimizing Computation Node

   (facet :initarg :facet :initform :exist :type (member :exist :input) :accessor tensor-facet)
   (named :initform :tensor :initarg :named :type keyword :accessor tensor-name)
   (input-shape :initarg :input-shape :initform nil :accessor tensor-input-shape)
   ))

(defmethod tensor-delete ((tensor AbstractTensor))
  nil
  )

;; Inline
(declaim (inline tensor-vec))
(defun tensor-vec (tensor)
  (declare (type AbstractTensor tensor)
	   (optimize (speed 3) (safety 0)))
  ;; add: (assert )
  (if (null (tensor-name tensor))
      (vec tensor)
      (if (vec tensor) ;; add: equal size?
	  (vec tensor)
	  (let ((alloc (make-tensor
			(shape tensor)
			:dtype (dtype tensor)
			:order (order tensor))))
	    (setf (tensor-vec tensor) (vec alloc))
	    (vec tensor)))))

(defun (setf tensor-vec) (new-value tensor)
  (declare (type AbstractTensor tensor))
  (write-vec new-value tensor))

;; (defmacro variable  ())
;; (defmacro parameter ())

(defmethod initialize-instance :after ((tensor AbstractTensor) &rest initargs &key &allow-other-keys)
  (let ((scalar-p   (getf initargs :scalar-p))
	(view       (getf initargs :view))
	(order      (getf initargs :order))
	(orig-shape (getf initargs :shape)))
    
    (setf (slot-value tensor 'orig-shape) orig-shape)
    (setf (slot-value tensor 'projected-p) (getf initargs :projected-p))
    
    (cond
      ((eql (getf initargs :facet) :input)
       (setf (tensor-stride tensor) (calc-strides orig-shape order))
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
	   (ignore vec))
  (if (typep shape-or-scalar 'list)
      (make-instance (car *using-backend*)
		     :dtype dtype
		     :order order
		     :requires-grad requires-grad
		     :shape shape-or-scalar
		     :projected-p nil
		     :facet :exist
		     :view view)
      (make-instance 'ScalarTensor
		     :scalar-p t
		     :vec shape-or-scalar
		     :shape nil
		     :dtype dtype
		     :requires-grad requires-grad
		     :projected-p nil
		     :facet :exist
		     :view view)))

;; It is allowed: (make-input `(batch-size 512))
(defun make-input (shape
		   named
		   &key
		     (dtype :float)
		     (order :column))
  "named ... variable-name (keyword)"
  (declare (type list shape)
	   (type (or null keyword) named))
  (make-instance (car *using-backend*)
		 :dtype dtype
		 :order order
		 :shape shape
		 :input-shape shape
		 :named (or named (symbol-name (gensym "ChainTMP")))
		 :facet :input))

(defun mref (tensor &rest subscripts)
  "Read-only. Only used for printing the tensor.
Whether you cares about performance or not, this function shouldn't be used ignoring for printing tensors."
  (declare (type list subscripts))
  (assert (eql (tensor-facet tensor) :exist)
	  nil
	  "Can't reference tensors which doesn't have a existing vec.")
  (vref tensor
	(apply #'+
	       (map 'list
		    #'(lambda (stride s view shape)
			(* stride (compute-stepby (subscript-view view)) (+ s (compute-visible-start-idx (subscript-view view) shape))))
		    (tensor-stride tensor)
		    subscripts
		    (tensor-view tensor)
		    (slot-value tensor 'orig-shape)))))

;; Note that mref is super slow and only used in a limited situation.
(defun (setf mref) (new-value tensor &rest subscripts)
  (declare (type list subscripts))
  
  (assert (eql (tensor-facet tensor) :exist)
	  nil
	  "Can't reference tensors which doesn't have a existing vec.")

  (setf (vref tensor
	      (apply #'+
		     (map 'list
			  #'(lambda (stride s view shape)
			      (* stride (compute-stepby (subscript-view view)) (+ s (compute-visible-start-idx (subscript-view view) shape))))
			  (tensor-stride tensor)
			  subscripts
			  (tensor-view tensor)
			  (slot-value tensor 'orig-shape))))
	new-value))

;; If you've created a new backend with different ptr, only you have to do is to define vref.
(defmethod vref ((tensor AbstractTensor) index)
    "Only used for printing the tensor.
Whether you cares about performance or not, this function shouldn't be used ignoring for printing tensors.

If you've created a new backend with having different ptr-type (can't be accessed by aref), only you have to do is to redefine vref."
  (declare (type fixnum index))
  (assert (eql (tensor-facet tensor) :exist)
	  nil
	  "Can't reference tensors which doesn't have a existing vec.")
  (aref (tensor-vec tensor) index))

(defmethod (setf vref) (new-value (tensor AbstractTensor) index)
    "Only used for printing the tensor.
Whether you cares about performance or not, this function shouldn't be used ignoring for printing tensors.

If you've created a new backend with having different ptr-type (can't be accessed by aref), only you have to do is to redefine vref."
  (declare (type fixnum index))
  (assert (eql (tensor-facet tensor) :exist)
	  nil
	  "Can't reference tensors which doesn't have a existing vec.")
  (setf (aref (tensor-vec tensor) index) new-value))

(defun embody-actual-tensor (input-tensor actual-tensor)
  "Moves actual-tensor(ExistTensor) -> input-tensor(InputTensor). (Pointers are shared.)"
  (declare (type AbstractTensor input-tensor actual-tensor))

  (assert (eql (tensor-facet input-tensor) :input)
	  nil
	  "Assertion Failed with (eql (tensor-facet input-tensor) :input)")

  (assert (eql (tensor-facet actual-tensor) :exist)
	  nil
	  "Assertion Failed with (eql (tensor-facet actual-tensor) :exist)")

  (setf (tensor-vec input-tensor) (tensor-vec actual-tensor)
	(slot-value input-tensor 'orig-shape) (slot-value actual-tensor 'orig-shape)
	
	(slot-value input-tensor 'visible-shape)
	(compute-visible-shape
	 (slot-value actual-tensor 'orig-shape)
	 (tensor-view actual-tensor)))
  t)


;; TODO: Extend the backward states.
(defun view (tensor &rest subscripts)
  "TODO: Docstring"
  ;; TODO: When tensor is scalar, return error.

  (assert (not (typep tensor 'ScalarTensor))
	  nil
	  "Assertion Failed with (not (typep tensor 'ScalarTensor))
ViewObject will never created to ScalarTensor.
got: ~a" tensor)

  (make-instance (car *using-backend*)
		 :dtype (dtype tensor)
		 :order (order tensor)
		 :requires-grad (slot-value tensor 'requires-grad)
		 :shape (slot-value tensor 'orig-shape)
		 :projected-p t
		 :past-view (tensor-view tensor)
		 :view subscripts
		 :input-shape (tensor-input-shape tensor)
		 :facet (tensor-facet tensor)
		 :named (tensor-name tensor)
		 :vec (slot-value tensor 'vec)))

;; TODO: Print ScalarTensor
(defmethod print-object ((tensor AbstractTensor) stream)
  (format stream
	  "{~a[~(~a~)] ~a ~a ~a
  ~a
  :facet :~(~a~)
  :requires-grad ~a
  :backward ~a}"
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
	  (let ((state (tensor-state tensor)))
	    (if state
		(format nil "~%  :vec-state [~(~a~)]" (statecontainer-state state))
		""))
	  (if (eql (tensor-facet tensor) :input)
	      (format nil "<<Not-Embodied ~a Tensor>>" (shape tensor))
	      ;; TODO: View -> View for printing 3d, 4d... tensor.
	      (render-tensor tensor :indent 2))
	  (tensor-facet tensor)
	  (slot-value tensor 'requires-grad)
	  (tensor-backward tensor)))

