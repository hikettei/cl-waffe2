
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

(defclass AbstractTensor ()
  ((nodes :initarg :nodes :initform nil :reader tensor-nodes :type list)

   ;; MultiDimensional APIs
   (orig-shape :initarg :shape :initform nil :reader original-shape :type list)
   (stride :initform nil :accessor tensor-stride :type list)
   (visible-shape :initform nil :reader shape :accessor tensor-visible-shape :type list)
   (view :initarg :view :initform nil :accessor tensor-view :type list)

   ;; Viewed?
   (projected-p :initarg :projected-p :initform nil :type boolean :reader tensor-projected-p)

   ;; Is it scalar?
   (scalar-p :initarg :scalar-p :initform nil)

   ;; vec container
   (vec :initarg :vec :initform nil :reader vec :writer write-vec)
   (dtype :initform :float :initarg :dtype :reader dtype)

   ;; Building Computation Nodes
   (backward  :initform nil :accessor tensor-backward)
   (state     :initform nil :accessor tensor-state)
   (variables :initform nil :accessor tensor-variables)

   (tensor-id :initform (gensym "TID") :accessor tensor-id)
   (nth-value :initform 0 :accessor tensor-out-n :type fixnum)

   (grad :initform nil :reader grad :writer set-grad)
   (gradient-adder :accessor gradient-adder)
   (requires-grad :initform nil :initarg :requires-grad :reader requires-grad :type boolean)
   (ancestor-param-p :initarg :requires-grad :initform nil :accessor ancestor-param-p :type boolean)
   (order :initarg :order :initform :column :type (satisfies order-p) :accessor order)
   
   (tensor-n-ref :initform 0 :accessor tensor-n-ref :type fixnum) ;; For optimizing
   (tensor-already-traced :initform nil :accessor tensor-traced-p :type boolean)
   
   (facet :initarg :facet :initform :exist :type (member :exist :input) :accessor tensor-facet)
   (named :initform :tensor :initarg :named :type keyword :accessor tensor-name)
   
   (input-shape :initarg :input-shape :initform nil :accessor tensor-input-shape))
  (:documentation "The class AbstractTensor is a fundamental datatype of dealing with various kernel (e.g.: CPU, Metal, CUDA...).

The class provides the fundamental features following:
    1. Lazy-Evaluated Multi-Dimensional Matrix APIs, and accordingly stride APIs for column/row major.
    2. Multi-Dimensional Matrix Offsets (i.e.: View APIs).
    3. Recording What Functions were called in the previous computation. (To construct backward.)
    4. vec container
    5. Keep Gradients
    6. Input API
    7. Trace Informations for JIT to create well-optimized computation node.

Users can extend this class and define the brand-new Tensor's Dtype depending on their use.

See the examples to understand how this could be achieved at ./source/backends/lisp/tensor.lisp. or ./source/backends/cpu.
"))

(defmethod tensor-delete ((tensor AbstractTensor))
  ""
  nil
  )

;; Inline
(declaim (inline tensor-vec))
(defun tensor-vec (tensor)
  "The function tensor-vec has a multiple behaviours depending on the state of tensor.

1. If the given tensor is existing, or is input but allocated.
   Returns the given tensor's vec.

2. If the given tensor is Input, and still not yet being accessed.
   Allocates the new area for matrix, and set tensor's vec slot it.
   The allocated area of tensor is returned.

In a short words:

Basically, tensor-vec is a function where:
  Input  -> AbstractTensor
  Output -> The tensor's vec slot (depends on their kernel)

However, using this function allows us to optimize the timing of allocation.

Note that this function is inlined.
"
  (declare (type AbstractTensor tensor)
	   (optimize (speed 3) (safety 0)))
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

(defun make-gradient-adder (target shape)
  "Returns a instant-kernel to add the new-gradient to given target."
  (let ((out (make-input shape nil
			 :dtype (dtype target)
			 :order (order target))))
    (with-no-grad
      (multiple-value-bind (fw bw vars params) (build (cl-waffe2/vm.nodes:forward (cl-waffe2/base-impl:AddNode) (grad target) out))
	(declare (ignore bw vars params))
	#'(lambda (new-value)
	    (assert (equal (shape new-value) shape)
		    nil
		    "Attempted to add a new grad: ~a to ~a but failed due to shaping problems."
		    (shape new-value)
		    shape)
	    (embody-actual-tensor out new-value)
	    (funcall fw)
	    nil)))))

(defmethod initialize-instance :after ((tensor AbstractTensor) &rest initargs &key &allow-other-keys)
  (let ((scalar-p   (getf initargs :scalar-p))
	(view       (getf initargs :view))
	(order      (getf initargs :order))
	(orig-shape (getf initargs :shape)))

    ;; orig-shape = used to compute strides.
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
	 nil)))
    
    ;; Setup utils for collecting gradients.
    (when (getf initargs :requires-grad)
      (set-grad (make-tensor
		 (tensor-visible-shape tensor)
		 :dtype (getf initargs :dtype)
		 :requires-grad nil
		 :order (getf initargs :order))
		tensor)
      (setf (gradient-adder tensor)
	    (make-gradient-adder tensor (tensor-visible-shape tensor))))))

(defmethod add-grads ((tensor AbstractTensor) new-value)
  "tensor's gradient += new-value"
  (assert (slot-value tensor 'requires-grad)
	  nil
	  "Assertion Failed with requires-grad = t")
  (funcall (gradient-adder tensor) new-value))

;; (defmethod init-grads ())
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
		      (order :column)
		      (initial-element))
  "Refering a first-priority of  *using-backends* (that is, a car part) to know what kernel to use, the function make-tensor creates and allocate a new matrix.

Input:
    - shape-or-scalar (Any), set list (consisted of fixnum) here to create a matrix, otherwise the ScalarTensor is forcibly created.
    - requires-grad (Boolean) Set t to create gradient. (e.g.: the tensor is needed to be optimized.)
    - dtype (keyword) Set dtype you wanna use. See also: (Dtype API)
    - vec (Anything) If you wanna pass the make-instance to already-allocated matrix, use this parameter.
    - order (member :column :row) 
    - initial-element (Optional)

With regard to practical usage, the tutorials would be more helpful rather than this document."
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
		     :initial-element initial-element
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
  "Referring a first-priority of *using-backend* (i.e.: car part), the function make-input creates a InputTensor.
WIth regard to practical usage, visit my tutorial.

Input:
    - Shape [list] Consisted of Fixnum or Symbol.
    - named [keyword]
    - dtype (as it is)
    - order (as it is)"
  (declare (type list shape)
	   (type (or null keyword) named))
  (if (equal shape `(1))
      (make-instance 'ScalarTensor
		     :dtype dtype
		     :order order
		     :shape shape
		     :input-shape shape
		     :named (or named (symbol-name (gensym "ChainTMPScalar")))
		     :facet :input)
      (make-instance (car *using-backend*)
		     :dtype dtype
		     :order order
		     :shape shape
		     :input-shape shape
		     :named (or named (symbol-name (gensym "ChainTMP")))
		     :facet :input)))

(defun mref (tensor &rest subscripts)
  "Read-only. Only used for printing the tensor.
Whether you cares about performance or not, this function shouldn't be used ignoring for printing tensors."
  (declare (type list subscripts))
  (assert (not (null (vec tensor)))
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
  
  (assert (not (null (vec tensor)))
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
  (assert (not (null (vec tensor)))
	  nil
	  "Can't reference tensors which doesn't have a existing vec.")
  (aref (tensor-vec tensor) index))

(defmethod (setf vref) (new-value (tensor AbstractTensor) index)
    "Only used for printing the tensor.
Whether you cares about performance or not, this function shouldn't be used ignoring for printing tensors.

If you've created a new backend with having different ptr-type (can't be accessed by aref), only you have to do is to redefine vref."
  (declare (type fixnum index))
  (assert (not (null (vec tensor)))
	  nil
	  "Can't reference tensors which doesn't have a existing vec.")
  (setf (aref (tensor-vec tensor) index) new-value))

(defun embody-actual-tensor (input-tensor actual-tensor)
  "Moves actual-tensor(ExistTensor) -> input-tensor(InputTensor). (Pointers are shared.)"
  (declare (type AbstractTensor input-tensor actual-tensor))

  (assert (eql (tensor-facet input-tensor) :input)
	  nil
	  "Assertion Failed with (eql (tensor-facet input-facet) :input)")

  (assert (vec actual-tensor)
	  nil
	  "Assertion Failed because the given actual-tensor doesn't have a existing vec.")

  (setf (tensor-vec input-tensor) (tensor-vec actual-tensor)
	(slot-value input-tensor 'orig-shape) (slot-value actual-tensor 'orig-shape)
	
	(slot-value input-tensor 'visible-shape)
	(compute-visible-shape
	 (slot-value actual-tensor 'orig-shape)
	 (tensor-view actual-tensor)))
  t)

;; (defun reshape ())

(defun view (tensor &rest subscripts)
  "The function view creates a view of given tensor.
Note that the function *view* doesn't records ANY NODES, while the function *!view* does.

Subscripts can be following:

- fixnum
- (start stop)
- (start stop by)
- t
- (:indices ...)
- (:tflist ...)
- (:broadcast fixnum)

Note that view is only created for Tensors, not a Scalar.
"
  ;; TODO: When tensor is scalar, return error.

  (when (typep tensor 'ScalarTensor)
    (format t ":BROADCAST IS IGNORED (TODO)")
    (return-from view tensor))

  (make-instance (car *using-backend*)
		 :dtype (dtype tensor)
		 :order (order tensor)
		 :requires-grad (slot-value tensor 'requires-grad)
		 :shape         (slot-value tensor 'orig-shape)
		 :projected-p   t
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
	  (if (and (eql (tensor-facet tensor) :input)
		   (null (vec tensor)))
	      (format nil "<<Not-Embodied ~a Tensor>>" (shape tensor))
	      ;; TODO: View -> View for printing 3d, 4d... tensor.
	      (render-tensor tensor :indent 2))
	  (tensor-facet tensor)
	  (slot-value tensor 'requires-grad)
	  (tensor-backward tensor)))

