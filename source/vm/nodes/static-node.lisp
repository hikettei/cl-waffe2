
(in-package :cl-waffe2/vm.nodes)

;; TODO: ====================================================
;; In accordance with refactoring of defnode, :where, add this feature: checking the number of arguments, number of outputs reading :where. (At: forward :around)
;; せっかく:whereで関数宣言してるのに安全性関連の機能が貧弱すぎる・・・⇦self書き忘れとか検知する、エラー内容が普通にうざい
;; Add Backward tests to define-static-node
;; Memo: (call (StaticNode) (parameter (randn `(10 10)))) is n't working for backward
;; But (call (StaticNode) (!copy (parameter (randn `(10 10))))) is working.
;; Add Documents
;; Update documents
;; SoftmaxCrossEntropy微分する

;; ===========================================================
;; Static Composite ==========================================
;; ===========================================================

(defmodel (Save-For-Backward-Static (self)
	   :where (Place[~] Tensor[~] -> [~])
	   :on-call-> ((self place tensor)
		       (declare (ignore self))
		       (cl-waffe2/base-impl:!move place tensor :force t))))

(define-composite-function (Save-For-Backward-Static) move-and-save-for-backward-static)

;; ==========================================================


;; Parameters
(defparameter *under-composite-node-mode* 0 "Incf 1 when go deeper with compis-temode, If count>=2, save-for-backwards are ignored.")

(defparameter *composite-node-self-global* nil "Used by with-setting-save4bw, with-reading-save4bw")



;; Implementation of save-for-backward/read-save-for-backward
;; Both of them are called with save-for-backward function binded by with-composite-node-mode
(defun apply-set-save-for-backward (self name tensor)
  (declare (type AbstractNode self)
	   (type symbol name)
	   (type AbstractTensor tensor))

  ;; Is this calling of save-for-backward is reachable? by backward => If so, make a copy.
  (when (and (not (>= *under-composite-node-mode* 2))
	     (null *no-grad*))
    (let ((past-sv4bw (slot-value self name)))

      ;; Save For Backward hasn't created yet?
      (when (null past-sv4bw)
	;; Make clone and allocate
	(let ((place (cl-waffe2/vm.generic-tensor::make-clone tensor)))
	  ;; Do allocation of place
	  (tensor-vec place)
	  ;; Set it to the slot
	  (setf (slot-value self name) place)))

      ;; Move: Existing Save-For-Backward-Place <- The target tensor.
      (move-and-save-for-backward-static (slot-value self name) tensor)
      nil)))

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

;; ==================================================================

(defmacro define-static-node ((name
			       (self-name &rest constructor-args)
			       &key
				 (where nil)
				 (slots nil)
				 (out-scalar-p nil)
				 (save-for-backward-names nil)
				 (forward nil)
				 (backward nil)
				 (documentation ""))
			      &body constructor-body)
  "
## [macro] define-static-node

```lisp
(define-static-node ((name
                          (self-name &rest constructor-args)
			       &key
				 (where nil)
				 (slots nil)
				 (out-scalar-p nil)
				 (save-for-backward-names nil)
				 (forward nil)
				 (backward nil)
				 (documentation \"\"))
			      &body constructor-body))
```

Defines a differentiable AbstractNode, but its forward/backward is defined by statcically notation.

### Inputs

1. `save-for-backward-names` ... an list of save-for-backwards (e.g.: `:save-for-backward-names (x y)`)

2. `backward` ... should be this form: `((self dout) ... (values x.grad y.grad ...))`

### Example

This macro differs from other `defnode` series macros, because the definition can be used in the same way for defun.

```lisp
(define-static-node (Static-Sin (self)
             :where (A[~] -> OUT[~])
             :save-for-backward-names (x-input)
             :forward ((self x)
                       (print \"Hi :) the operation sin is executed.\")
                       (with-setting-save4bw ((x-input x))
                            (proceed (!sin x))))
             :backward ((self dout)
                (with-reading-save4bw ((x x-input))
                       (values (proceed (!mul (!cos x) dout)))))))
```

Calling `Static-Sin` but the `print` function is still not yet called.

```lisp
(call (Static-Sin) (randn `(3 3)))
{CPUTENSOR[float] :shape (3 3) :named ChainTMP22057 
  :vec-state [maybe-not-computed]
  <<Not-Embodied (3 3) Tensor>>
  :facet :input
  :requires-grad NIL
  :backward <Node: STATIC-SIN-T (A[~] -> OUT[~])>}
```

The moment someone accept the computation node and invoked it, `print` is called.

```lisp
(proceed *)

Hi :) the operation sin is executed.
{CPUTENSOR[float] :shape (3 3) :named ChainTMP22130 
  :vec-state [computed]
  ((-0.44811794 -0.8374244  -0.9075781)
   (-0.9591228  0.58454794  0.9774129)
   (-0.8381093  -0.36447936 0.9476587))
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
```

But what if one wants to save the given tensors for a future call of backward? Yes, to do this, the functions `set-save-for-backward` and `read-save-for-backward` is available. :)
"

  ;; TODO: Checking the number of arguments
  ;; TODO: set-save-for-backward
  ;; TODO: gradtmp space
  
  (let ((forward-args  (car forward))
	(backward-args (car backward))
	(forward-body  (cdr forward))
	(backward-body (cdr backward))
	(backward-node-name (symb name '-backward))

	(forward-flet-name (symb name '-forward-body))
	(backward-flet-name (symb name '-backward-body))
	(save-for-backward-slots
	  (loop for s in save-for-backward-names
		collect `(,s :initform nil :type (or null AbstractTensor)))))
    
    `(progn
       ;; Called only when backward-mode
       (define-and-impl-node (,backward-node-name (self forward-self)
			      :device t
			      ;; Temporary Restores Nodes used in forward.
			      :slots ((forward-self :initarg :forward-self :reader read-forward-self))
			      ;; forward -> backward => backward -> forward
			      :where ,(where->backward where)
			      :forward ((,@backward-args)
					;; Swapping (self args...) -> (forward-self args...)					
					
					(flet ((,backward-flet-name (,@backward-args)
						 (declare (ignorable ,(car backward-args)))
						 (with-composite-node-mode ,(car backward-args)
						   (locally ,@backward-body))))
					  (multiple-value-bind (,@backward-args)
					      (apply #'values (list (read-forward-self ,(car backward-args)) ,@(cdr backward-args)))
					    ;; FixME: Static Backward with multiple arguments?
					    `(funcall ,#',backward-flet-name ,,@backward-args))))))

       ;; This is a main part of composite-node.
       (define-and-impl-node (,name (,self-name ,@constructor-args)
			      :device t
			      :where ,where
			      :slots (,@slots
				      ,@save-for-backward-slots)
			      :out-scalar-p ,out-scalar-p
			      :documentation ,documentation
			      :forward ((,@forward-args)
					(flet ((,forward-flet-name (,@forward-args)
						 (declare (ignorable ,(car forward-args)))
						 (with-composite-node-mode ,(car forward-args)
						   (locally ,@forward-body))))
					  `(funcall ,#',forward-flet-name ,,@forward-args)))
			      :backward ((,@backward-args)
					 ;; Initializes backward node with (Backward self)
					 (forward (,backward-node-name ,(car backward-args)) ,@(cdr backward-args))))
	 ,@constructor-body))))


