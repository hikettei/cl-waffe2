
(in-package :cl-waffe2/base-impl)

;; [TODO] :
;;  ir.lisp provides nodes which controls the flow of nodes:
;;
;;   IfNode
;;   MapNode
;;   RecurrentNode
;;
;; Additionally,
;;  Lazy-Function
;;  Lazy-Reduce

;; [TODO] Enhancement:&rest argument for defnode

(deftype lazy-computable-t ()
  "A list of types which can be dynamically compiled and cached"
  `(or symbol function))

(defun all-bool-p (x) (every #'(lambda (x) (typep x 'boolean)) x))

(defnode (Lazy-Function-Node (self forward &key (backward nil) (sv4bw nil))
	  :slots ((forward   :initarg :forward  :initform nil :accessor forward-of)
		  (backward  :initarg :backward :initform nil :accessor backward-of)
		  (reduced-to :initform nil :accessor reduced-of))
	  :documentation "
An abstract computation node that dynamically compile the given kernel specified by `forward` with a loop, applying it to X and OUT element-wise. A backend `LispTensor` already provides a standard implementation of it and can be used by the `(cl-waffe2/base-impl:lazy ...)` function. This node is useful when calling mathematical functions not provided by cl-waffe2 as standard; (Note that no speed improvement can be expected from SIMD.)

```lisp
;; Example:
(lazy #'sin (randn `(3 3)) :diff #'cos)
```

### Inputs

- `forward[symbol or function]` indicates a name of function of forward propagation. the function must receive a single argument of corresponding element.

- `backward[symbol or function]` indicates a name of function of backward propagation. As the backward definition indicates, the gradient of the previous node is automatically combined by Lazy-Function-Node. therefore, #'cos is enough for example.

- `sv4bw[boolean]` set T to copy the result of X.

### Workload

- [x] implement
- [x] make it differentiable
- [x] compiled kernels are cached in LUT.
- [ ] parallelize by lparallel
- [ ] Loop Collapse/Reordering
"
	  :where (X[~] OUT[~] -> OUT[~])
	  :backward ((self dout x out)
		     (declare (ignore out))
		     (when (null (backward-of self))
		       (error "lazy: In order to differentiate the lazy operation ~a, specify :backward.
(lazy op tensor ... :diff nil)
                           L Specify this form."
			      (forward-of self)))
		     (values (!mul dout (lazy (backward-of self) x)) nil)))
  (setf (node-save-for-backward self) (list sv4bw nil)))

;; [TODO] Remove with LazyArangeNode
(defnode (Lazy-Index-Components-Node (self forward)
	  :slots ((forward   :initarg :forward  :initform nil :accessor forward-of)
		  (backward  :initform nil :accessor backward-of)
		  (reduced-to :initform nil :accessor reduced-of))
	  :where (X[~] OUT[~] -> OUT[~])))

(defnode (Lazy-Reduce-Node (self forward &key (backward nil) (sv4bw nil) (reduced 1))
	  :slots ((forward  :initarg :forward  :initform nil :accessor forward-of)
		  (backward :initarg :backward :initform nil :accessor backward-of)
		  (reduced  :initarg :reduced  :initform nil :accessor reduced-of))
	  :documentation "
As well as `Lazy-Function-Node`, this node dynamically compiles the given kernel specified by `forward` with a loop, applying it to X and OUT element-wise. The only difference is that the last dimension of returned tensor is reduced to `reduced`. The kernel function `forward` wil receive all elements of the last dimension of `X`, and selects from it and return `reduced` values. (Note that the value is returned by `(apply #'values list)`, NOT A LIST.)

See the example of `lazy-reduce`.

As of this writing, this node isn't differentiable.

### Workload

- [x] implement
- [ ] make it differentiable
- [ ] caching
- [ ] parallelize by lparallel
- [ ] loop oriented optimizations
"
	  :where (Reduced[~ reduced] X[~ dim] -> Reduced[~ reduced]))
  (setf (node-save-for-backward self) (list nil sv4bw)))

(declaim (ftype (function (lazy-computable-t
			   AbstractTensor
			   &key
			   (:diff (or null lazy-computable-t)))
			  AbstractTensor)
		lazy))
(defun lazy (op tensor &key (diff nil))
  "
## [function] lazy

```lisp
(lazy op tensor &key (diff nil))
```

Invokes AbstractNode `Lazy-Function-Node` that dynamically compile the given kernel specified by `op` with a loop, applying it to tensor and store the result to the copied one (this node can be pruned if unnecessary).

```lisp
;; Example:
(lazy #'sin (randn `(3 3)) :diff #'cos)
```
### Inputs

- `op[symbol or function]` indicates a name of function of forward propagation. the function must receive a single argument of corresponding element.

- `tensor[AbstractTensor]` a tensor to be applied.

- `diff[symbol or function]` indicates a name of function of backward propagation. As the backward definition indicates, the gradient of the previous node is automatically combined by Lazy-Function-Node. therefore, #'cos is enough for example. If diff is set to something, `save-for-backward` automaticaly becomes T.
"
  (declare (type AbstractTensor tensor))

  (let ((out
	  (call (Lazy-Function-Node op :backward diff :sv4bw (when diff t))
		tensor
		(!copy tensor :maybe-in-place t))))
    ;; Returning the first of returned tensors
    out))

(defun lazy-index-components (op tensor)
  (call (Lazy-Index-Components-Node op) tensor (!copy tensor :maybe-in-place t)))

(declaim (ftype (function (lazy-computable-t
			   AbstractTensor
			   &key
			   (:reduce-to t)
			   (:axis fixnum)
			   (:diff (or null lazy-computable-t)))
			  AbstractTensor)
		lazy-reduce))
(defun lazy-reduce (op tensor &key (reduce-to 1) (axis -1) (diff nil) &aux (base-permute (wf/t::tensor-permute-order tensor)))
  "
## [function] lazy-reduce

```lisp
(lazy-reduce op tensor &key (reduce-to 1) (diff nil))
```

(See also: Lazy-Reduce-Node)

As well as `lazy`, this function dynamically compiles the given kernel specified by `op` with a loop, applying it to tensor and stores the result to the copied tensor (can be pruned if unnecessary). The only difference is that the last dimension of returned tensor is reduced to `reduced`. The kernel function `op` wil receive all elements of the last dimension of `X`, and selects from it and return `reduced` values. (Note that the value is returned by `(apply #'values list)`, NOT A LIST.)

### Input


- `op[symbol or function]` indicates a name of function of forward propagation. the function will receive all elements of last dimension.

- `tensor[AbstractTensor]` tensor to be applied.

- `reduced-to[fixnum]` a fixnum indicating the number of elements reduced.

- `diff[symbol or function]` (currently ignored)

- `sv4bw[bool]` (currently ignored)


### Examples

`my-topk` is a function to retrieve the Kth largest value in the last dimension.

```lisp
(defun topk (k)
  #'(lambda (&rest args)
      (let ((topN (sort args #'>)))
	(apply #'values (loop for n upfrom 0 below K collect (nth n topN))))))

(lazy-reduce (topk 3) (ax+b `(10 10) 1 0) :reduce-to 3)

(defun my-topk (tensor k)
    (lazy-reduce (topk K) tensor :reduce-to k))

(my-topk (ax+b `(10 10) 1 0) 3)
```

```lisp
(lazy-reduce #'max (ax+b `(10 10) 1 0))
```
"
  (declare (type AbstractTensor tensor))

  (let ((axis (if (< axis 0)
		  (+ axis (length (shape tensor)))
		  axis)))

    (when (not (= axis (1- (length (shape tensor)))))
      (let ((last  (1- (length (shape tensor))))
	    (order (copy-list base-permute)))
	(rotatef (nth last order) (nth axis order))
	(setf tensor (apply #'!permute tensor order)))))

    (assert (null diff)
	    nil
	    "Assertion Failed: As of this writing, lazy-reduce isn't differentiable... (TODO)
Set :diff = nil to ignore this assertion")
    
    (let* ((reduced (make-input `(,@(butlast (shape tensor)) ,(wf/vm:make-lazyaxis reduce-to)) nil
				:dtype (dtype tensor)
				:order (order tensor)))
	   (out
	     (call (Lazy-Reduce-Node op :backward diff :sv4bw (when diff t) :reduced (wf/vm:make-lazyaxis reduce-to))
		   reduced
		   tensor)))
      ;; Returning the first of returned tensors
      (when (not (= axis (1- (length (shape tensor)))))
	(let ((last  (1- (length (shape tensor))))
	      (order (copy-list base-permute)))
	  (rotatef (nth last order) (nth axis order))
	  (setf out (apply #'!permute out order))))
      out))

