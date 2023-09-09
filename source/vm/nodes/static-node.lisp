
(in-package :cl-waffe2/vm.nodes)

;; TODO: ====================================================
;; In accordance with refactoring of defnode, :where, add this feature: checking the number of arguments, number of outputs reading :where. (At: forward :around)
;; Add Backward tests to define-static-node
;; Memo: (call (StaticNode) (parameter (randn `(10 10)))) is n't working for backward
;; But (call (StaticNode) (!copy (parameter (randn `(10 10))))) is working.

;; ===========================================================
;; Static Composite ==========================================
;; ===========================================================

;; Parameters
(defparameter *under-composite-node-mode* 0 "Incf 1 when go deeper with compis-temode, If count>=2, save-for-backwards are ignored.")

(defparameter *composite-node-self-global* nil "Used by with-setting-save4bw, with-reading-save4bw")




;; Using proceed instead composite-function is now much wiser desicion.

;; Implementation of save-for-backward/read-save-for-backward
;; Both of them are called with save-for-backward function binded by with-composite-node-mode
(defun apply-set-save-for-backward (self name tensor)
  (declare (type AbstractNode self)
	   (type symbol name)
	   (type AbstractTensor tensor))

  ;; Is this calling of save-for-backward is reachable? by backward => If so, make a copy.

  (when (and (not (>= *under-composite-node-mode* 2))
	     (not *no-grad*))
    (let ((past-sv4bw (slot-value self name)))

      ;; Save For Backward hasn't created yet?
      (when (null past-sv4bw)
	;; Make clone and allocate
	;; [FIXME] Is this tensor creation gc-reachable??
	(let ((place (cl-waffe2/vm.generic-tensor::make-clone-exist tensor)))
	  ;; Do allocation of place
	  ;; Set it to the slot
	  (setf (slot-value self name) place)))

      (cl-waffe2/vm::%vm-move (slot-value self name) tensor)

      ;; FixME

      (when (and (scalar-p (slot-value self name))
		 (scalar-p tensor))
	(setf (tensor-vec (slot-value self name)) (cl-waffe2/vm.generic-tensor::vec tensor)))))
  nil)

(defun apply-read-save-for-backward (self name)
  (declare (type AbstractNode self)
	   (type symbol name))
  (let ((place (slot-value self name)))
    (typecase place
      (AbstractTensor place)
      (T
       (if (null place)
	   (error "apply-read-save-for-backward: Reading the value of ~a, ~a returned nil.
This is because in forward process, set-save-for-backward never called." self name)
	   (error "apply-read-save-for-backward: Reading the value of ~a, ~a returned the wrong value: ~a.

It should be AbstractTensor."
		  self
		  name
		  place))))))


(defmacro with-composite-node-mode (self &body body)
  "Under the body execution, Composite-Node-Mode is enabled which is:

1. save-for-backward/read-save-for-backward function become available.
2. *under-composite-node-mode* becomes t."
  `(let ((*under-composite-node-mode* (1+ *under-composite-node-mode*))
	 (*composite-node-self-global* ,self))
     (flet ((set-save-for-backward (self name tensor)
	      (apply-set-save-for-backward self name tensor))
	    (read-save-for-backward (self name)
	      (apply-read-save-for-backward self name)))
       
       ;; Instead of ignore:
       #'set-save-for-backward
       #'read-save-for-backward
       
       ,@body)))


;; ==================================================================
;; Exports
;; ==================================================================

;; Binding a set-save-for-backward function at toplevel
(defun set-save-for-backward (self name tensor)
  "
## [function] set-save-for-backward

```lisp
(set-save-for-backward self name tensor)
```

The function `set-save-for-backward` saves the given `tensor` to the `name` slot of self for a future call of backward.

This function is dedicated to the macro `define-static-node`, so it should be placed at the forward/backward definition of the macro, otherwise, the wrong function is binded which returns simple-error. In addition, The place to save the tensor, should be also declared in `:save-for-backward-names` in the `define-static-node` macro.

Note that this function is ignored in specific conditions: `*no-grad*` is t or `set-save-for-backward` in the forward definition in the forward definition. (i.e.: the place which is never called.)

See also: `read-save-for-backward` `with-setting-sv4bw` `with-reading-sv4bw` `define-static-node`
"
  
  (error "set-save-for-backward: Attempted to call (set-save-for-backward ~a ~a ~a), but failed. This is because the function isn't placed under `(with-composite-node-mode)` macro.

That is: `set-save-for-backward` is available only when called under the forward/backward definition of `(define-static-node)`. (as long as you're not doing anything weird.)

If you're working with `defnode` or `defmodel` and needs save-for-backward features,
`:save-for-backward` keyword would help you :).
"
	 self name tensor))

;; Binding a read-save-for-backward function at toplevel
(defun read-save-for-backward (self name)
  "
## [function] read-save-for-backward

```lisp
(read-save-for-backward self name)
```

Reading the slot of `name` in `self`, the function `read-save-for-backward` returns a saved tensor by `set-save-for-backward`.

For the same reason of `set-save-for-backward`, this function should be placed at right place.
"
  (error "read-save-for-backward: Attempted to call (read-save-for-backward ~a ~a), but failed. This is because the function isn't placed under `(with-composite-node-mode)` macro.

That is: `read-save-for-backward` is available only when called under the forward/backward definition of `(define-static-node)`. (as long as you're not doing anything weird.)

If you're working with `defnode` or `defmodel` and needs save-for-backward features,
`:save-for-backward` keyword would help you :).
"
	 self name))

(defmacro with-reading-save4bw ((&rest input-forms) &body body)
  "
## [macro] with-reading-save4bw

```lisp
(with-reading-save4bw ((&rest input-forms) &body body))

input-form = (variable-place save-for-backward-name)
```

Reading the save-for-backward of currently working node, the macro binds each `variable-place` the stored tensor.
"
  (labels ((expand-forms (rest-forms)
	     (if (car rest-forms)
		 `(let ((,(caar rest-forms) (read-save-for-backward *composite-node-self-global* ',@(cdar rest-forms))))
		    ,(expand-forms (cdr rest-forms)))
		 `(progn ,@body))))
    (expand-forms input-forms)))

(defmacro with-setting-save4bw ((&rest input-forms) &body body)
  "
## [macro] with-setting-save4bw

```lisp
(with-setting-save4bw ((&rest input-forms) &body body))
input-form = (save-place tensor)
```

Saves the given tensors to save-place, in the currently working node.
"
  `(progn
     ,@(map 'list #'(lambda (input-form)
		      `(set-save-for-backward *composite-node-self-global* ',(car input-form) ,@(cdr input-form)))
	    input-forms)
     ,@body))

(defmacro define-op ((name
		      (self &rest constructor-args)
		      &key
			(where nil)
			(slots nil)
			(out-scalar-p nil)
			(save-for-backward-names nil)
			(forward nil)
			(backward nil)
			(extends nil)
			(documentation ""))
		     &body constructor-body)
  "
## [macro] define-op

`define-op` = `defnode` + `define-impl-op`

Defines a differentiable AbstractNode which its definition is given by a function.

```lisp
(define-op (name (self &rest constructor-args) where slots out-scalar-p save-for-backward-names forward backward documentation extends) &body body)
```

### Effects

This macro defines:

1. two `AbstractNodes` named `name` and `name-backward` (if backward is given)

### Example

```lisp
(define-op (TestAdd-Scalar (self)
	    :where (A[scal] B[scal] -> A[scal] where scal = 1)
            :out-scalar-p t
	    :forward ((self a b)
		      (make-tensor
		       (+ (tensor-vec a)
			  (tensor-vec b))))
	    :backward ((self dy)
		       (values dy dy))))
```
"
  (declare (type list save-for-backward-names))
  (when (null where)
    (error "define-op: The where declaration isn't provided.
            (define-op (name (self ...)
                             :where (A[i j] -> B[i j])
                                └── Fill this form
                    ...))"))

  (multiple-value-bind (fw-in bw-in fw-state bw-state let-binding) (parse-subscript where)
    (declare (ignore let-binding fw-state bw-state bw-in))

    (let* ((forward-args  (car forward))
	   (backward-args (car backward))
	   (forward-body  (cdr forward))
	   (backward-body (cdr backward))
	   	
	   (backward-node-name (symb name '-backward))
	   (save-for-backward-slots
	     (loop for s in save-for-backward-names
		   collect `(,s :initform nil :type (or null AbstractTensor)))))

      (when (not (= (length forward-args) (1+ (length fw-in))))
	(error "define-op: The number of arguments in forward doesn't match with declared in :where.
:where -> ~a
butgot -> ~a"
	       fw-in
	       forward-args))

      (when (and backward
		 (not (= (length backward-args) 2)))
	(error "define-op: The number of arguments in backward should be 2.
                (define-op (name (self ...)
                                 ...
                                 :backward ((self dy)
                                              └── Check here
                                            ...
                                            (values ...))))"))

      (with-gensyms (fw-self dout in-shape out-shape)
	(setq in-shape (intern (symbol-name in-shape) *package*))
	(setq out-shape (intern (symbol-name out-shape) *package*))
	`(progn
	   ;; Declares AbstractNode
	   (defnode (,name (,self ,@constructor-args)
		     :where ,where
		     :out-scalar-p ,out-scalar-p
		     :slots (,@slots
			     ,@save-for-backward-slots)
		     :extends ,extends
		     :documentation ,documentation)
	     ,@constructor-body)
	   
	   (defnode (,backward-node-name (,self ,fw-self ,in-shape ,out-shape)
		     ;; where : DOUT X [BW-SHAPES[0]] Y -> ... X Y...
		     :out-scalar-p ,out-scalar-p
		     :where (,dout [,(symb dout '-size)]
				   ,@(loop for fw-tensor-name in fw-in
					   for n fixnum upfrom 0
					   append
					   `(,fw-tensor-name [,(symb 'in-shapes n)]))
				   ->
				   ,@(loop for fw-tensor-name in fw-in
					   for n fixnum upfrom 0
					   append
					   `(,(symb fw-tensor-name '-grad) [,(symb 'in-shapes n)]))
				   where
				   ,(symb dout '-size)
				   =
				   (nth 0 ,out-shape)
				   ,@(loop for fw-tensor-name in fw-in
					   for n fixnum upfrom 0
					   append
					   `(,(symb 'in-shapes n) = (nth ,n ,in-shape))))
		     :slots ((fw-self :initform nil))
		     :extends ,extends)
	     (setf (slot-value ,self 'fw-self) ,fw-self
		   (ignore-shape-error ,self) t))
	   
	   (define-impl-op (,name :device t)
			   :forward ((,@forward-args)
				     (with-composite-node-mode ,(car forward-args)
				       ,@forward-body))
			   
			   :backward ((,self ,dout ,@(cdr forward-args))
				      (let ((,in-shape (map 'list #'shape (list ,@(cdr forward-args))))
					    (,out-shape (node-out-sizes ,self)))
					(call
					 (,backward-node-name ,self ,in-shape ,out-shape)
					 ,dout
					 ,@(cdr forward-args)))))

	   (define-impl-op (,backward-node-name :device t)
			   :forward ((,@backward-args ,@(cdr forward-args)) ;; self dout x y ...
				     (let ((,(car backward-args) (slot-value ,(car backward-args) 'fw-self)))
       				       (with-composite-node-mode ,(car backward-args)
					 ,@backward-body)))))))))


