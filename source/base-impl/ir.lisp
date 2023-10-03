
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

;; [TODO]
;;  - LispTensor Implementation (OK)
;;  - Docstring
;;  - Add: topk (OK, standard implはexportしない)
;;  - Add: Tests
;;  - Update: Mkdocs
;;  - Add: define-elwise-impl

(deftype lazy-computable-t ()
  "A list of types which can be dynamically compiled and cached"
  `(or symbol list function))

(defun all-bool-p (x) (every #'(lambda (x) (typep x 'boolean)) x))
(deftype sv4bw-t () ;; = (T NIL T NIL ...)
  `(or null (and list (satisfies all-bool-p))))

(defnode (Lazy-Function-Node (self forward &key (backward nil) (sv4bw nil))
	  :slots ((forward   :initarg :forward  :initform nil :accessor forward-of)
		  (backward  :initarg :backward :initform nil :accessor backward-of)
		  (reduced-to :initform nil :accessor reduced-of))
	  :documentation "

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
  (setf (node-save-for-backward self) sv4bw))

(defnode (Lazy-Reduce-Node (self forward &key (backward nil) (sv4bw nil) (reduced 1))
	  :slots ((forward  :initarg :forward  :initform nil :accessor forward-of)
		  (backward :initarg :backward :initform nil :accessor backward-of)
		  (reduced  :initarg :reduced  :initform nil :accessor reduced-of))
	  :documentation "
"
	  :where (Reduced[~ reduced] X[~ dim] -> Reduced[~ reduced]))
  (setf (node-save-for-backward self) sv4bw))

(declaim (ftype (function (lazy-computable-t
			   AbstractTensor
			   &key
			   (:diff (or null lazy-computable-t))
			   (:sv4bw sv4bw-t))
			  AbstractTensor)
		lazy))
(defun lazy (op tensor &key (diff nil) (sv4bw nil))
  "
## [function] lazy

```lisp
(lazy op tensor &key (diff nil) (sv4bw nil))
```

### Input

"
  (declare (type AbstractTensor tensor)
	   (type list sv4bw))

  (let ((out
	  (call (Lazy-Function-Node op :backward diff :sv4bw sv4bw)
		tensor
		(!copy tensor :maybe-in-place t))))
    ;; Returning the first of returned tensors
    out))


(declaim (ftype (function (lazy-computable-t
			   AbstractTensor
			   &key
			   (:reduce-to fixnum)
			   (:diff (or null lazy-computable-t))
			   (:sv4bw sv4bw-t))
			  AbstractTensor)
		lazy-reduce))
(defun lazy-reduce (op tensor &key (reduce-to 1) (diff nil) (sv4bw nil))
  "
## [function] lazy-reduce

```lisp
(lazy-reduce op tensor &key (reduce-to 1) (diff nil) (sv4bw nil))
```

### Input

"
  (declare (type AbstractTensor tensor)
	   (type list sv4bw))

  (assert (null diff)
	  nil
	  "Assertion Failed: As of this writing, lazy-reduce isn't differentiable... (TODO)
Set :diff = nil to ignore this assertion")
  
  (let* ((reduced (make-input `(,@(butlast (shape tensor)) ,reduce-to) nil
			      :dtype (dtype tensor)
			      :order (order tensor)))
	 (out
	   (call (Lazy-Reduce-Node op :backward diff :sv4bw sv4bw :reduced reduce-to)
		 reduced
		 tensor)))
    ;; Returning the first of returned tensors
    out))

