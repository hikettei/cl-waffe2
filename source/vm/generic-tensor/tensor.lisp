
(in-package :cl-waffe2/vm.generic-tensor)

;;
;; 
;; AbstractTensor | Multiple Backends
;;
;;

(defparameter *using-backend*
  `()
  "cl-waffe searches for computation nodes in the following order and uses the first one it finds. (Priority1 Priority2 ...)
Default: `(cl-waffe2/vm.generic-tensor:CPUTensor)
PriorityN must be a subclass of cl-waffe2/vm.generic-tensor:AbstractTensor")


;; make-tensor under the forward method IS PROHIBITED because cl-waffe2 isn't designed so.
;; If developers want to detect runtime creation of tensors, set *detect-runtime-creation-tensor*=t and debug it.
(defparameter *runtime-mode-p* nil)
(defparameter *detect-runtime-creation-tensor* nil "For debugging, set t to detect the runtime creation of tensors, which may worse performance.")

(defparameter *default-dtype* :float  "")
(defparameter *default-order* :column "")

(defparameter *with-printing-tensor-omitted* nil "If t, the displayed tensor is omitted")

(defun order-p (name)
  (declare (type keyword name))
  (or (eql name :column) (eql name :row)))

(defun find-scalar-tensor ()
  (or (find 'ScalarTensor *using-backend*
	    :test #'(lambda (x y) (subtypep y x)))
      'ScalarTensor))

;; orig-shape    ... The shape of storage vec.
;; visible-shape ... The viewed shape
;; actual-shape  ... visible-shape but broadcasting is ignored.

(defclass AbstractTensor ()
  ((orig-shape :initarg :shape :initform nil :reader original-shape :type list)
   (stride :initform nil :accessor tensor-stride :type list)
   (permute-order :initform nil :initarg :permute-order :accessor tensor-permute-order :type list)
   (visible-shape :initform nil :reader shape :accessor tensor-visible-shape :type list)
   (view :initarg :view :initform nil :accessor tensor-view :type list)

   ;; Viewed?
   (projected-p :initarg :projected-p :initform nil :type boolean :reader tensor-projected-p)

   ;; Is it scalar?
   (scalar-p :initarg :scalar-p :initform nil :reader scalar-p)
   (detach-p :initform nil :accessor detach-p)
   ;; vec container
   (vec :initarg :vec :initform nil :reader vec :writer write-vec)
   (dtype :initform :float :initarg :dtype :reader dtype)

   (offset :initform 0 :accessor tensor-initial-offset)

   ;; Building Computation Nodes
   (backward  :initform nil :accessor tensor-backward)

   ;; Storing the result of compiling
   (state     :initform nil :accessor tensor-state)

   ;; Records previous variables
   (variables :initform nil :accessor tensor-variables)


   ;; tensor-id  ... indicates which pointer to use or copied?, plus, the index in the mempool.
   ;; tensor-iid ... used for topological sorting

   (lock-id-p :initform nil :accessor tensor-id-lock-p)
   (tensor-id :initform (gensym "TID") :type symbol :accessor tensor-id)         
   (tensor-ident-id :initform (gensym "TIDi") :accessor tensor-iid)
   
   (nth-value :initform 0 :accessor tensor-out-n :type fixnum)

   ;; Optimizing
   (optimizer :initform nil :accessor tensor-optimizer :type (or null cl-waffe2/optimizers:AbstractOptimizer))

   (grad :initform nil :reader grad :writer set-grad)
   (grad-count :initform 0 :type fixnum :accessor tensor-grad-count)
   
   (save-for-backward-space       :initform nil :accessor save-for-backward-space)
   
   (requires-grad :initform nil :initarg :requires-grad :reader requires-grad :type boolean)
   (ancestor-param-p :initarg :requires-grad :initform nil :accessor ancestor-param-p :type boolean)
   
   (order :initarg :order :initform :column :type (satisfies order-p) :accessor order)
   
   (flexible-p :initform nil :accessor tensor-flexible-p :type (or boolean fixnum))
   
   (facet :initarg :facet :initform :exist :type (member :exist :input) :accessor tensor-facet)
   (named :initform :tensor :initarg :named :type keyword :accessor tensor-name)

   (allocate-time-state :initform nil :type (or null Adjustable-Shape-State) :accessor tensor-alloc-state)
   (protect-me :initform nil :accessor tensor-protect-me) ;; If t, cache never ignored.
   (input-shape :initarg :input-shape :initform nil :reader tensor-input-shape))
  (:documentation "
## [class] AbstractTensor

AbstractTensor is a CLOS class that Wraps existing data structures such as matrices in an abstract class in automatic differential programming using cl-waffe2, and further adds information about computation nodes, gradients, etc.

Tensors can be created by the `make-tensor` function.

```lisp
(make-tensor `(3 3))
```

Plus, InputTensors (lazy-evaluated tensors), which is used to delay allocation timing, to use dynamic shaping, and to store the result, can be created by the `make-input` function.

```lisp
(make-input `(3 3) :A) ;; Set :A=nil to register as a temporary space.
```

As an applied use, users can create new `AbstractTensor` that inherit from AbstractTensor. In addition, inheriting existing AbstractTensors (e.g.: `LispTensor` for CL Standard Array) allows reusing descriptions such as allocations.

```lisp
(defclass MyOriginalTensor (AbstractTensor) nil)
(defclass MyCPUTensor      (LispTensor) nil)
```

Declare the priority of the device to be used with the with-devices macro.

```lisp
;; Higher <-> Lower
(with-devices (MyCPUTensor MyOriginalTensor CPUTensor)
    (make-tensor `(10 10)))
```

All available devices can be accessed with the `(show-backends)` function, and they can only be used as devices together if they are shown to have an inheritance relationship.

If a completely new Tensor is defined from AbstractTensor, cl-waffe2 can handle it completely in a fast form by writing the following additional information.

- Allocator: `initialize-instance :before method`

- Storage Accessor: `vref` and `(setf vref)` method

- Finalizer: `tensor-finalizer` method

- (Optional) Backend State: `current-backend-state` method

- (Optional) a `cl-waffe2/vm:defpath` macro to enable device-specific optimization.

This is the simplest case of `MyTensor` which works on CL Standard Array.

```lisp
(defclass MyTensor (AbstractTensor) nil)

;; Allocators satisfy the following properties
;; 1. When facet is not `:exist`, do nothing.
;; 2. If `vec` is specified as an argument, use this, and do not allocate any tensors.
;; 3. Otherwise, allocate the tensor with:
;;     1. Dtype -> :dtype
;;     2. Size  -> :shape (must be 1D on the memory)
;;     3. initial-element -> :initial-element
(defmethod initialize-instance :before ((tensor MyTensor)
					&rest initargs
					&key &allow-other-keys)
  (let* ((shape (getf initargs :shape))
	 (dtype (dtype->lisp-type (getf initargs :dtype)))
	 (vec   (getf initargs :vec))
	 (facet (getf initargs :facet))
	 (initial-element (coerce (or (getf initargs :initial-element) 0) dtype)))
    (when (eql facet :exist)
      (if vec
	  (setf (tensor-vec tensor) vec)
	  (setf (tensor-vec tensor)
		(make-array
		 (apply #'* shape)
		 :element-type dtype
		 :initial-element initial-element))))))

;; vref reads the index th element of storage vec, this is must be a setfable.
;; Leave the annoying and complicated stride/offset computations to cl-waffe2!
(defmethod vref ((tensor MyTensor) index)
  (declare (type fixnum index))
  (aref (tensor-vec tensor) index))

(defmethod (setf vref) (new-value (tensor MyTensor) index)
  (declare (type fixnum index))
  (setf (aref (tensor-vec tensor) index) new-value))


;; The method should return a lambda function, if its storage vector isn't gc-reachable.
;; Finalizers are called when quitting (with-memory-pool ...) macro.
(defmethod tensor-finalizer ((tensor MyTensor))
    ;; Returning a dummy finalizer
    #'(lambda ()))

;; The function (show-backends) will display all devices and their information
;; If you want to put something, override this method and return a string.
(defmethod current-backend-state ((backend-name (eql 'MyTensor)))
  \"Hello This is an demo\")

;; For FusionOp and defpath macro usage, see the :cl-waffe2/vm docs.
```

`MyTensor` is now recognised as a usable device, so operations can be defined using the define-impl and define-impl-op macros.


### [function] shape

`(shape tensor)` returns a visible shape of the given tensor.

### [function] dims

`(dims tensor)` returns a rank of the given tensor.

### [function] total

`(total tensor)` returns the number of total visible elements of the giventensor.

### [slot] orig-shape (List)

stores the shape of storage vec.

### [accessor] initial-offset (fixnum)

stores the offset of the tensor. In default, set to 0. Shape testing, for example, does not work, so use with caution.

`(tensor-initial-offset tensor)`

### [slot] stride (list)

`(tensor-stride tensor)` stores the stride of tensor.

### [slot] visible-shape (list)

`(shape tensor)`

### [slot] view (list)

Returns a list of ViewInstruction, created by the function `(view tensor ...)` or `(!view tensor ...)` to create a backward.

`(tensor-view tensor)`

### [slot] projected-p (boolean)

Set t if `(apply #'* orig-shape) == (apply #'* visible-shape)` otherwise set nil.

If t, the tensor is created by `!view` or `view` functions.

### [slot] scalar-p

Set t if the tensor should be represented as a scalar. In cl-waffe2, it's not a pretty thing but scalars are represented as a `(apply #'* shape)=1` tensors. ranks are anything but for the most case, returns 1.

### [slot] detach-p

Set T to detach the tensor at a certain position.

### [slot] state

(tensor-state tensor) stores `StateContainer`.

### [slot] variables

`(tensor-variables tensor)` stores the previous variables if the tensor is created by any operation.

### [slot] tensor-id (symbol)

Indicates where the Tensor is stored, (e.g. in a virtual machine). In-place operations inherit tensor-id from variables called with, and should not be used for topological sorting.

### [slot] tensor-iid (symbol)

It holds an ID that is guaranteed to be absolutely unique to the processing system generated by gensym. Used for topological sorting.

### [slot] grad (AbstractTensor)

If the tensor is created by (parameter ...) or with `:requires-grad=t`, `(grad tensor)` will return a gradient.

### [slot] backward (AbstractNode)

`(tensor-backward tensor)` returns a abstractnode if the tensor is created by any operation.

### [slot] requires-grad (Boolean)

Set T to hold the gradients.

### [slot] ancestor-param-p (Boolean)

Set T if compilers can reach any tensors with `:requires-grad=t`, by tracing the tensor.

### [slot] flexible-p (Fixnum or Null)

Indicates the position of broadcastable axis.

### [slot] facet (keyword)

AbstractTensors in cl-waffe2 has a two state: `ExistTensor` and `InputTensor`. `ExistTensor` is a just tensor with allocated storage vec, made by make-tensor function. On the other hand InputTensor is a lazy-evaluated tensor, allocation won't be done until it is needed.

:exist to ExitTensor, :input to InputTensor.

### [method] mref

`(mref tensor &rest subscripts)` will reads a cetrain position of storage vec. This is setfable. In terms of performance, it is much faster way to edit a storage vec that using `(change-facet)` function and convert into other forms.

### Hooking Optimizers and Optimizing Parameters

(TODO)


"))


(defgeneric tensor-finalizer (tensor))

(defun dummy-finalizer ())
(defmethod tensor-finalizer ((tensor AbstractTensor))
  (if (next-method-p)
      (call-next-method)
      #'dummy-finalizer))

(defgeneric current-backend-state (backend-name)
  (:documentation "
## [generic] current-backend-state

```lisp
(current-backend-state backend-name)
```

The generic function current-backend-state is used to rendering (show-backends) function.
"))

(defmethod current-backend-state ((backend-name t))
  (if (next-method-p)
      (call-next-method)
      "No status."))

(defun total (tensor)
  (declare (type AbstractTensor tensor))
  (apply #'lazy-mulup (shape tensor)))

(defun dims (tensor)
  (declare (type AbstractTensor tensor))
  (length (shape tensor)))

(defun permuted-p (tensor)
  (declare (type AbstractTensor)
	   (optimize (speed 3)))
  (let ((a (copy-list (tensor-permute-order tensor))))
    (not (equal (sort a #'>) (tensor-permute-order tensor)))))

;; Inline
(declaim (inline tensor-vec))
(defun tensor-vec (tensor)
  "

## [function] tensor-vec

```lisp
(tensor-vec tensor)
```

If the given tensor is a ExistTensor, returns its storage vec.

If the given tensor is a InputTensor, allocates the area for tensor and return its storage vec.

This function is setfable and inlined.
"
  (declare (type AbstractTensor tensor))

  ;; tensor-vec is called under the build execution:

  (let ((out
	  (cond
	    ;; Adjustable Table is allowed to use:
	    ((and *static-alloc-state* (cl-waffe2/vm::tensor-tmp-p tensor))
	     (cl-waffe2/vm::storage-vec-from-memory-pool *static-alloc-state* tensor))
	    (T (or (vec tensor) (get-from-global-memory-pool tensor))))))
    (if (scalar-p tensor)
	(if (lazy-variable-p out)
	    (read-lazy-var out)
	    out)
	out)))

(defun (setf tensor-vec) (new-value tensor)
  (declare (type AbstractTensor tensor))
  (write-vec new-value tensor))

;; Initializes generic uis of tensors.
(defmethod initialize-instance :after ((tensor AbstractTensor) &rest initargs &key &allow-other-keys)
  (let ((scalar-p    (getf initargs :scalar-p))
	(view        (getf initargs :view))
	(order       (getf initargs :order))
	(orig-shape  (getf initargs :shape))
	(create-from (getf initargs :create-from)))

    (when *detect-runtime-creation-tensor*
      (when *runtime-mode-p*
	(warn "Detected runtime creation of tensors.")))

    ;; create-from   = extend permute information from the tensor create-from.
    ;; orig-shape    = used to compute strides, always synchronized with vec.
    ;; visible-shape = visible size for users, always modified by making a view.
    
    (setf (slot-value tensor 'orig-shape)      orig-shape)
    (setf (slot-value tensor 'projected-p)     (getf initargs :projected-p))
    (setf (slot-value tensor 'ancestor-param-p) (ancestor-param-p tensor)) ;; Is it worth to call backward?

    (when create-from
      (setf (tensor-initial-offset tensor) (tensor-initial-offset create-from)))
    
    ;; Updates/Initializes: Strides/Shapes/View/Permution Informations
    (cond
      ((and create-from
	    (permuted-p create-from)
	    (equal orig-shape (original-shape create-from))
	    (not (scalar-p tensor)))
       ;; Subject to shuffle: orig-shape visible-shape view permute-order stride
       ;; Never Do this: Recomputing strides, USE create-from

       ;; MEMO: copy-list is needed??
       (setf (tensor-stride tensor) (copy-list (tensor-stride create-from))
	     (slot-value tensor 'orig-shape) (copy-list (original-shape create-from))
	     (tensor-permute-order tensor) (copy-list (tensor-permute-order create-from))
	     
	     (tensor-view tensor) (parse-view-subscripts tensor (getf initargs :past-view) (or view `(t)))
	     (tensor-visible-shape tensor) (compute-visible-shape orig-shape (tensor-view tensor))))
      ((eql (getf initargs :facet) :input)
       (when (not scalar-p)
	 (setf (tensor-stride tensor) (calc-strides orig-shape order))
	 (setf (tensor-view tensor)
	       (parse-view-subscripts tensor (getf initargs :past-view) (or view `(t))))
	 (setf (tensor-visible-shape tensor)
	       (compute-visible-shape orig-shape (tensor-view tensor)))
	 nil))
      (T
       (when (not scalar-p)
	 (setf (tensor-stride tensor) (calc-strides orig-shape order))
	 (setf (tensor-view tensor) (parse-view-subscripts tensor (getf initargs :past-view) (or view `(t))))
	 (setf (tensor-visible-shape tensor)
	       (compute-visible-shape orig-shape (tensor-view tensor)))
	 nil)))
    
    ;; Initial Permution: 5 4 3 2 ... 1 0
    (unless (tensor-permute-order tensor)
      (setf (tensor-permute-order tensor)
	    (loop for i downfrom (length (shape tensor)) to 1
		  collect (1- i))))

    ;; If needed, update information about gradients
    (when (slot-value tensor 'requires-grad)
      (if (scalar-p tensor)
	  (set-grad (make-tensor 0
				 :dtype (dtype tensor)
				 :requires-grad nil)
		    tensor)
	  (set-grad (make-tensor
		     (tensor-visible-shape tensor)
		     :dtype (getf initargs :dtype)
		     :requires-grad nil
		     :order (getf initargs :order))
		    tensor)))))

(defun transfer-vec-information (from to)
  "Transfer information that makes a vec vec"
  (declare (type AbstractTensor from to))
  (setf (slot-value to 'orig-shape) (slot-value from 'orig-shape)
	(slot-value to 'visible-shape) (slot-value from 'visible-shape)
	(tensor-stride to) (tensor-stride from)
	(tensor-view to) (tensor-view from)
	(tensor-permute-order to) (tensor-permute-order from)
	;;(slot-value to 'input-shape) (slot-value from 'input-shape)
	)
  nil)

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

(defun make-tensor-from-vec (shape dtype vec
			     &key (order *default-order*))
  (make-instance (car *using-backend*)
		 :dtype dtype
		 :order order
		 :requires-grad nil
		 :shape (copy-list shape)
		 :projected-p nil
		 :vec vec
		 :facet :exist))

(defun make-tensor (shape-or-scalar
		    &key
		      (requires-grad nil)
		      (dtype *default-dtype*)
		      (view nil)
		      (order *default-order*) ;; TODO (retain-grads nil)
		      (initial-element)
		      (device nil)
		      (create-from nil))
  "
## [function] make-tensor

```
(make-tensor shape-or-scalar
	       &key
		  (requires-grad nil)
		  (dtype *default-dtype*)
		  (view nil)
		  (order *default-order*)
		  (initial-element nil)
                  (device nil))
```

Created a new ExistTensor of a device of `(car *using-backend*)`.

### Inputs

1. `shape-or-scalar`[Anything] If set to list, creates a new matrix. Otherwise (e.g.: set to fixnum), creates a ScalarTensor. In that case, cl-waffe2 uses the highest priority device from `*using-backends*` parameter that inherits from the `ScalarTensor` class.

2. `requires-grad`[Boolean] Set t to holds a gradients. `(parameter tensor)` will also do the same work. Under `(with-no-grad ...)` macro. This is set to nil forcibly.

3. `dtype`[keyword] Set keyword indicating a type of elements.

4. `order`[keyword] set keyword indicating the order of elments from `:column` or `:row`. in default set to `:column`.

5. `initial-element`[Anything] Set anything which you want to set as a initial element.

6. `device[symbol or null]` If set to symbol, the function returns with making a tensor of device.
"
  (declare (type list view))
  (if (typep shape-or-scalar 'list)
      (make-instance (or device (car *using-backend*))
		     :dtype dtype
		     :order order
		     :create-from create-from
		     :requires-grad requires-grad
		     :shape (copy-list shape-or-scalar)
		     :projected-p nil
		     :facet :exist
		     :initial-element initial-element
		     :view view)
      (make-instance (or device (find-scalar-tensor))
		     :scalar-p t
		     :vec (coerce-lazy shape-or-scalar (dtype->lisp-type dtype))
		     :shape nil
		     :dtype dtype
		     :create-from create-from
		     :requires-grad requires-grad
		     :projected-p nil
		     :facet :exist
		     :view view)))

;; It is allowed: (make-input `(batch-size 512))
(defun make-input (shape
		   named
		   &key
		     (create-from nil)
		     (scalar-p nil)
		     (dtype *default-dtype*)
		     (order *default-order*))
  "
## [function] make-input

```lisp
(make-input shape named &key (created-from nil) (scalar-p nil) (dtype *default-dtype*) (order *default-order*))
```

Creates a new InputTensor. The allocation won't be done until the function `(tensor-vec tensor)` is called. In cl-waffe2, InputTensors can be applied for various things, for example, tracing the structure of computation node, used as a temporary tensor which can be pruned later by a compiler, as an argument of the computation node compiled by the `build` function. 

### Inputs

`Shape` [list] Set the shape of tensor. You can also use symbols if shapes can be changed later. The function `set-input` will update all symbols declared in the computation node, and accordingly, strides/shapes etc... will be also updated to minimise compiling-time overhead (use `build` and `forward` to do this). ScalarTensors aren't created by setting it=`<<Something but not a list>>`. Instead, set `scalar-p=t`.

`Named` [keyword or null] Indicates the name of tensor. If set to keyword, This means the name of the argument when compiled into a function, which can be changed later. If set to nil, the name is filled with `gensym` indicating the index in the memory-pool.

`scalar-p` [boolean] Set t to create a scalar.

`dtype` [keyword] Set dtype.

`order` [keyword] Set order.

`create-from[nil or AbstractTensor]` The returned InputTensor will extend Permutions/Strides and so on from `create-from` if any.
"
  (declare (type list shape)
	   (type (or null keyword) named))
  (make-instance (if scalar-p
		     (find-scalar-tensor)
		     (car *using-backend*))
		 :scalar-p scalar-p
		 :create-from create-from
		 :dtype dtype
		 :order order
		 :shape shape
		 :input-shape shape
		 :named (or named (symbol-name (gensym "ChainTMP")))
		 :facet :input))

(defun mref (tensor &rest subscripts)
  "The function mref is only used to print/initialize tensors, accessing the index of subscripts **with** considering views..

If you cares about performance, dont' use `mref`, but `!view`.

This function is setfable."
  (declare (type list subscripts))
  (assert (not (null (vec tensor)))
	  nil
	  "Can't reference tensors which doesn't have an existing vec.")
  (vref tensor
	(+
	 (tensor-initial-offset tensor)
	 (apply #'+
		(map 'list
		     #'(lambda (stride s view shape)
			 (declare (ignore shape))
			 (* stride (compute-stepby (subscript-view view))
			    (+ s (compute-visible-start-idx (subscript-view view)))))
		     (tensor-stride tensor)
		     subscripts
		     (tensor-view tensor)
		     (slot-value tensor 'orig-shape))))))

;; Note that mref is super slow and only used in a limited situation.
(defun (setf mref) (new-value tensor &rest subscripts)
  (declare (type list subscripts))
  
  (assert (not (null (vec tensor)))
	  nil
	  "Can't reference tensors which doesn't have an existing vec.")

  (setf (vref tensor
	      (+
	       (tensor-initial-offset tensor)
	       (apply #'+
		      (map 'list
			   #'(lambda (stride s view shape)
			       (declare (ignore shape))
			       (* stride (compute-stepby (subscript-view view))
				  (+ s (compute-visible-start-idx (subscript-view view)))))
			   (tensor-stride tensor)
			   subscripts
			   (tensor-view tensor)
			   (slot-value tensor 'orig-shape)))))
	new-value))

;; If you've created a new backend with different ptr, only you have to do is to define vref.
(defmethod vref ((tensor AbstractTensor) index)
  "`vref` is a generic-function to access the `vec` slot of specific backends tensor, and returns `index`th element on `vec` slot **without** considering views.

If you added a new backend with having different ptr-type (can't be accessed by aref), override this method and `(setf vref)`.

### Example

```lisp
(defmethod vref ((tensor YourBackend) index)
    (aref (tensor-vec tensor) index))

(defmethod (setf vref) (new-value (tensor YourBackend) index)
    (setf (aref (tensor-vec tensor) index) new-value))
```
"
  (declare (type fixnum index))
  (assert (not (null (vec tensor)))
	  nil
	  "Can't reference tensors which doesn't have an existing vec.")
  (aref (tensor-vec tensor) index))

(defmethod (setf vref) (new-value (tensor AbstractTensor) index)
    "An setfable version of vref."
  (declare (type fixnum index))
  (assert (not (null (vec tensor)))
	  nil
	  "Can't reference tensors which doesn't have an existing vec.")
  (setf (aref (tensor-vec tensor) index) new-value))

;; input <- actual
(defun embody-actual-tensor (input-tensor actual-tensor)
  "Moves actual-tensor(ExistTensor) -> input-tensor(InputTensor). (Pointers are shared.)"
  (declare (type AbstractTensor input-tensor actual-tensor)
	   (optimize (speed 3)))

  
  ;;(assert (eql (tensor-facet input-tensor) :input)
  ;;	  nil
  ;;	  "Assertion Failed with (eql (tensor-facet input-facet) :input)")
  
  (assert (vec actual-tensor)
	  nil
	  "Assertion Failed because the given actual-tensor doesn't have an existing vec.")

  (when (or (numberp (vec input-tensor))
	    (numberp (vec actual-tensor)))
    (setf (tensor-vec input-tensor) (tensor-vec actual-tensor))
    (return-from embody-actual-tensor t))

  (let ((actual-tensor
	  (if (and (= (the fixnum (dims actual-tensor)) (the fixnum (dims input-tensor)))		   
		   (permuted-p input-tensor)
		   (not (equal (shape input-tensor) (shape actual-tensor))))
	      (apply #'permute* actual-tensor (tensor-permute-order input-tensor))
	      actual-tensor)))
    
    (setf (tensor-vec input-tensor) (tensor-vec actual-tensor)
	  (slot-value input-tensor 'orig-shape) (slot-value actual-tensor 'orig-shape)
	  (tensor-permute-order input-tensor) (tensor-permute-order actual-tensor)
	  (tensor-view input-tensor) (tensor-view actual-tensor)
	  (tensor-stride input-tensor) (tensor-stride actual-tensor)
	  (tensor-visible-shape input-tensor) (tensor-visible-shape actual-tensor)
	  (tensor-initial-offset input-tensor) (tensor-initial-offset actual-tensor)
	  (slot-value input-tensor 'projected-p) (slot-value actual-tensor 'projected-p)))
  t)

(defun embody-tensor-vec (input-tensor actual-tensor)
  "Moves actual-tensor(ExistTensor) -> input-tensor(InputTensor) but shape/strides"
  (declare (type AbstractTensor input-tensor actual-tensor))

  (assert (vec actual-tensor)
	  nil
	  "embody-tensor-vec: Assertion Failed because the given actual-tensor doesn't have an existing vec.")

  (when (or (scalar-p input-tensor)
	    (scalar-p actual-tensor))
    (setf (tensor-vec input-tensor) (vec actual-tensor))
    (return-from embody-tensor-vec t))

  (when (or (numberp (vec input-tensor))
	    (numberp (vec actual-tensor)))
    (setf (tensor-vec input-tensor) (vec actual-tensor))
    (return-from embody-tensor-vec t))

  ;; Offsets?
  (let ((actual-tensor
	  ;; V delete?
	  (if (and (= (the fixnum (dims actual-tensor)) (the fixnum (dims input-tensor)))
		   (permuted-p input-tensor))
	      (apply #'permute* actual-tensor (tensor-permute-order input-tensor))
	      actual-tensor)))

    (setf (tensor-vec input-tensor) (tensor-vec actual-tensor)
	  (tensor-initial-offset input-tensor) (tensor-initial-offset actual-tensor)
	  (slot-value input-tensor 'orig-shape) (translate-adjustable-shape (original-shape actual-tensor))
	  (tensor-permute-order input-tensor) (tensor-permute-order actual-tensor)
	  (tensor-view input-tensor) (tensor-view actual-tensor)
	  (tensor-visible-shape input-tensor) (translate-adjustable-shape (tensor-visible-shape actual-tensor))
	  (slot-value input-tensor 'projected-p) (slot-value actual-tensor 'projected-p)))
  t)

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

  (let ((subscripts
	  (loop for s in subscripts collect (force-list s))))
    (when (subtypep (class-name (class-of tensor)) 'ScalarTensor)
      (if (and subscripts
	       (not (every #'(lambda (x) (eql t x)) subscripts)))
	  (format t ":BROADCAST IS IGNORED for scalar input (TODO)"))
      (return-from view tensor))

    (make-instance (car *using-backend*)
		   :create-from tensor
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
		   :vec (vec tensor))))

(defun detach-and-clone (tensor)
  (make-instance (car *using-backend*)
		 :create-from tensor
		 :scalar-p (scalar-p tensor)
		 :dtype (dtype tensor)
		 :order (order tensor)
		 :shape (slot-value tensor 'orig-shape)
		 :projected-p t
		 :past-view (tensor-view tensor)
		 :input-shape (tensor-input-shape tensor)
		 :facet (tensor-facet tensor)
		 :named (tensor-name tensor)
		 :vec (vec tensor)))

(defun detach-and-clone1 (tensor)
  "Set tensor-info-save-p = t"
  (make-instance (car *using-backend*)
		 :create-from tensor
		 :scalar-p (scalar-p tensor)
		 :dtype (dtype tensor)
		 :order (order tensor)
		 :shape (copy-list (slot-value tensor 'orig-shape))
		 :projected-p t
		 ;; If you have encounter unknown view error related to view, consider thining :view of detach-and-clone
		 :view (copy-list (map 'list #'force-list (tensor-view tensor)))
		 :past-view (copy-list (tensor-view tensor))
		 :input-shape (copy-list (tensor-input-shape tensor))
		 :facet (tensor-facet tensor)
		 :named (tensor-name tensor)
		 :vec (vec tensor)
		 :slot-info-copied t))

(defun sync-permute! (tensor)
  (declare (type AbstractTensor tensor))
  (macrolet ((apply-permute (accessor tensor)
	       `(loop with copy = (copy-list ,accessor)
		      with rank = (1- (length (shape ,tensor)))
		      for o in (tensor-permute-order ,tensor)
		      for kth upfrom 0
		      do (let ((pos (- rank o)))
			   (setf (nth kth ,accessor)
				 (nth pos copy))))))
    (apply-permute (tensor-stride tensor) tensor)
    (apply-permute (tensor-view tensor) tensor)
    (apply-permute (slot-value tensor 'visible-shape) tensor)
    (apply-permute (slot-value tensor 'orig-shape) tensor)
    
    (when (and (slot-value tensor 'input-shape)
	       ;; [Bug] Sometime, The rank of InputShape and its actual shape does not match. However in that case, input shape is no longer needed. so ignore it.
	       (= (dims tensor) (length (tensor-input-shape tensor))))
      (apply-permute (slot-value tensor 'input-shape) tensor))
    nil))

(defun permute-computable-p (old-order new-order)
  (equal (sort (copy-list old-order) #'<)
	 (sort (copy-list new-order) #'<)))

(defun apply-flexible-subscript (old-orders new-orders position)
  (let* ((goal-length (length old-orders))
	 (diff (- goal-length (length new-orders))))
    (if (and (> diff 0) position)
        `(,@(loop for i upfrom 0 below position
		  collect (nth i new-orders))
	  ,@(loop for i upfrom position below (+ position diff)
		  collect (nth i old-orders))
	  ,@(loop for i upfrom (+ position diff) below goal-length
		  collect (let ((i (- i diff)))
			    (nth i new-orders))))
	new-orders)))

(defun permute* (tensor &rest orders)
  "
Shuffles the order of axes of the tensor.

:~ to make it flexible.

(E.g.: 1 :~ 5)

(TODO Document)

Initial permute order: n n-1 n-2 ... 3 2 1.

The axis is computed at Order[N]th loop.

Ex: transposing the tensor.
```
(permute* tensor :~ 0 1)
```

Initial value of permute: n n-1 ... 2 1 0.

See also: `!permute`
"

  (when (scalar-p tensor)
    (error "permute*: permutes of array can only created for tensors, not a scalar."))
  
  (when (> (count :~ orders) 1)
    (error "permute*: The keyword :~~ must be appeared at once.: ~a" orders))

  (let* ((tensor-new  (detach-and-clone1 tensor)) ;; Detaching from computation nodes by making a view of T T T....
	 (old-orders (tensor-permute-order tensor))
	 (pure-orders (remove :~ orders)) ;; order consisted of fixnum
	 (new-orders
	   (loop for rank fixnum upfrom 0 below (length (shape tensor))
		 for order in pure-orders
		 collect (progn
			   (when (and (numberp order)
				      (null (nth order old-orders)))
			     (error "permute*: the number of order is out of range: ~a" order))
			   order)))
	 (new-orders (if (position :~ orders)
			 (apply-flexible-subscript
			  old-orders
			  pure-orders
			  (position :~ orders))
			 new-orders)))
    (if (not (permute-computable-p old-orders new-orders))
	(error "permute*: before and after the operation, all axes must be used: ~a -> ~a." old-orders new-orders))

    (setf (tensor-permute-order tensor-new) new-orders)
    
    ;; shuffle strides.
    (sync-permute! tensor-new)
    
    tensor-new))

(defun detach! (tensor)
  "detach tensor from computation node."
  (setf (detach-p tensor) t)
  tensor)

(defun parameter (tensor)
  "
## [function] parameter

```
(parameter tensor)
```

Creates a new tensor with :requires-grad=t from the given tensor. If the tensor is remained to be computed, parameter will use the result from `proceed`.

### Example

```lisp
(parameter (randn `(3 3)))
```
"
  
  (declare (type AbstractTensor tensor))
  (let ((out (if (tensor-backward tensor)
		 (cl-waffe2/base-impl:proceed tensor)
		 tensor)))
    (setf (tensor-facet out) :exist)
    (setf (slot-value out 'requires-grad) t)
    (setf (tensor-name out) nil)
    (if (scalar-p tensor)
	(make-tensor (tensor-vec tensor)
		     :requires-grad t
		     :dtype (dtype tensor)
		     :order (order tensor))
	;; detach from computation node.
	(view out))))


(defun render-shape (tensor)
  "Returns a shape"
  (let ((flexible-p (tensor-flexible-p tensor))
	(shape      (shape tensor)))
    (with-output-to-string (str)
      (format str "(")
      (loop for i upfrom 0
	    for s in shape
	    if (and flexible-p
		    (= i flexible-p))
	      do (format str "<1 x N> ")
	    do (format str "~a" s)	    
	    unless (= i (1- (length shape))) ;; unless the last
	      do (format str " "))
      
      (when (and flexible-p
		 (= (length shape) flexible-p))
	(format str " <1 x N>"))
      (format str ")"))))

(defun state-name (tensor state)
  (declare (type StateContainer state))
  (let ((f-exist-p (statecontainer-forward-result state))
	(b-exist-p (statecontainer-backward-result state)))
    (cond
      ((or (subtypep (class-of (tensor-backward tensor))
		     'cl-waffe2/base-impl::ProceedNode)
	   (statecontainer-latest-p state))
       :computed)
      ((and (null f-exist-p)
	    (null b-exist-p))
       :maybe-not-computed)
      ((and (null b-exist-p))
       :wait-for-backward)
      (T
       :wait-for-reset))))

(defmethod print-object ((tensor AbstractTensor) stream)

  (when *with-printing-tensor-omitted*
    (format stream "<<~a Tensor (Omitted)>>" (shape tensor))
    (return-from print-object))
  
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
		 (format nil ":shape ~a" (render-shape tensor)))
		(T
		 ;; It has a view
		 (format nil ":shape ~a -> :view ~a -> :visible-shape ~a"
			 (slot-value tensor 'orig-shape)
			 (slot-value tensor 'view)
			 (render-shape tensor)))))
	  (if (eql (tensor-facet tensor) :input)
	      (if (keywordp (tensor-name tensor))
		  (format nil ":named :~a" (tensor-name tensor))
		  (format nil ":id ~a"     (tensor-id tensor)))
	      "")
	  (let ((state (tensor-state tensor)))
	    (if state
		;; TODO: Update vec-state
		(format nil "~%  :vec-state [~(~a~)]" (state-name tensor state))
		""))
	  (if (and (eql (tensor-facet tensor) :input)
		   (null (vec tensor)))
	      (format nil "  <<Not allocated: size=~a>>" (shape tensor))
	      ;; TODO: View -> View for printing 3d, 4d... tensor.
	      (render-tensor tensor :indent 2))
	  (if (and (eql (tensor-facet tensor) :input)
		   (not (keywordp (tensor-name tensor))))
	      (format nil "input~%  :belongs-to :memory-pool")
	      (tensor-facet tensor))
	  (slot-value tensor 'requires-grad)
	  (tensor-backward tensor)))

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  APIs for save_for_backward (cl-waffe2 VM, internal usage)
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(defun system-lazy-set-save-for-backward (tensor)
  (if (save-for-backward-space tensor)
      nil
      (let ((tensor-clone (make-clone tensor nil)))
	(setf (tensor-id-lock-p tensor-clone) t)
	(setf (save-for-backward-space tensor) tensor-clone)
	tensor)))

(defun system-lazy-read-save-for-backward (tensor)
  (save-for-backward-space tensor))

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  Optimazations
;;
;;  Hook Optimizer -> Call Optimizer -> Reset-Grad
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(defun hook-optimizer! (tensor optimizer)
  "
## [function] hook-optimizer!

```lisp
(hook-optimizer! tensor optimizer)
```

Hooks the optimizer to the tensor.

### Inputs

tensor[AbstractTensor]

optimizer[AbstractOptimizer]

"
  (declare (type AbstractTensor tensor)
	   (type cl-waffe2/optimizers:AbstractOptimizer optimizer))
  (when (slot-value tensor 'requires-grad)
    (setf (tensor-optimizer tensor) optimizer)))

(defun call-optimizer! (tensor)
  "
## [function] call-optimizer!

```lisp
(call-optimizer! tensor)
```

Reading the `(grad tensor)`, the function invokes the optimizer hooked to the tensor.
"
  (declare (type AbstractTensor))
  (when (slot-value tensor 'requires-grad)
    (when (null (tensor-optimizer tensor))
      (error "The tensor ~a has no optimizer hooked. Call (hook-optimizer! tensor optimizer) in advance."
	     tensor))
    (cl-waffe2/optimizers:step-optimize (tensor-optimizer tensor))))

(defun reset-grad! (tensor)
  "
## [function] reset-grad!

Resets the gradient of the tensor with zero.
"
  (declare (type AbstractTensor tensor))
  (when (slot-value tensor 'requires-grad)
    (if (scalar-p tensor)
	(setf (tensor-vec tensor) (make-tensor 0 :dtype (dtype tensor)))
	nil)))

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(defun shape-with-broadcastable (tensor)
  "
## [function] shape-with-broadcastable

Returns shape but <1 x N> parts are replaced with -1.
"
  (declare (type AbstractTensor tensor))
  (let ((flexible-p (tensor-flexible-p tensor))
	(shape      (shape tensor))
	(out))

    (loop for i upfrom 0
	  for s in shape
	  if (and flexible-p
		  (= i flexible-p))
	    do (push -1 out)
	  do (push s out))
    
    (when (and flexible-p
	       (= (length shape) flexible-p))
      (push -1 out))
    (reverse out)))

(defun tensor-actual-stride (tensor)
  (declare (type AbstractTensor tensor))
  ;; Returns a stride of tensor, but broadcasted axis=0
  (loop for s in (tensor-stride tensor)
	for v in (tensor-view tensor)
	if (eql (viewtype (force-list v)) :broadcast)
	  collect 0
	else
	  collect s))

(defun tensor-memory-size (tensor)
  (declare (type AbstractTensor tensor))
  (flet ((->size (dtype)
	   (cffi:foreign-type-size dtype)))
    (if (dtype->lisp-type (dtype tensor))
	(if (scalar-p tensor)
	    (->size (dtype tensor))
	    (if (some #'symbolp (original-shape tensor))
		(let ((total-size)
		      (sym))
		  (dolist (s (original-shape tensor))
		    (if (numberp s)
			(push s total-size)
			(push s sym)))
		  (setq total-size (apply #'* total-size))
		  (setq sym (reverse sym))
		  (values (->size (dtype tensor))
			  total-size
			  sym))
		(* (->size (dtype tensor))
		   (apply #'* (original-shape tensor)))))
	(progn
	  (warn "tensor-memory-size: Unknown dtype ~a is accumlated as 0" (dtype tensor))
	  0))))

