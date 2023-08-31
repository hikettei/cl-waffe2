
(in-package :cl-waffe2/vm.nodes)

;; TODO: ====================================================
;; In accordance with refactoring of defnode, :where, add this feature: checking the number of arguments, number of outputs reading :where. (At: forward :around)
;; Add Backward tests to define-static-node
;; Memo: (call (StaticNode) (parameter (randn `(10 10)))) is n't working for backward
;; But (call (StaticNode) (!copy (parameter (randn `(10 10))))) is working.

;; ===========================================================
;; Static Composite ==========================================
;; ===========================================================

(defmodel (Save-For-Backward-Static (self)
	   :where (Place[~] Tensor[~] -> [~])
	   :on-call-> ((self place tensor)
		       (declare (ignore self))
		       (cl-waffe2/base-impl:!move place tensor :force t))))

;; (defmodel-as ...
;;(define-composite-function (Save-For-Backward-Static) move-and-save-for-backward-static)

;; ==========================================================


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
	(let ((place (cl-waffe2/vm.generic-tensor::make-clone tensor)))
	  ;; Do allocation of place
	  (tensor-vec place)
	  ;; Set it to the slot
	  (setf (slot-value self name) place)))

      ;; Move: Existing Save-For-Backward-Place <- The target tensor.
      ;;(move-and-save-for-backward-static (slot-value self name) tensor)
      (cl-waffe2/base-impl:proceed (cl-waffe2/base-impl:!move (slot-value self name) tensor :force t))

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


;; [TODO] define-static-node ... 削除する！！！
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

Defines a differentiable AbstractNode, but its forward/backward definition is given by a function.

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
	(result-tmp (gensym))

	(save-for-backward-slots
	  (loop for s in save-for-backward-names
		collect `(,s :initform nil :type (or null AbstractTensor)))))
    
    `(progn
       ;; Called only when backward-mode
       (define-and-impl-node (,backward-node-name (self forward-self)
			      :device t
			      ;; Temporary Restores Nodes used in forward.
			      :cache-when-compiled t
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
					    ;; (setf self.backward-result = result)
					    `(let ((,',result-tmp (multiple-value-list (funcall ,#',backward-flet-name ,,@backward-args))))
					       (setf (backward-result ,,(car backward-args))
						     (list ,,@(loop for i upfrom 0 below (length (cdr forward-args)) collect nil)))
					       
					       ;; backward-result = (AbstractTensor AbstractTensor) x n_input_args
					       ;; nil isn't allowed

					       ;; (car backward-args) = self
					       ;; nil = set forward args for a while,
					       ;; the node pruned later.
					       (loop for arg in (cdr (forward-args ,,(car backward-args)))
						     for nth fixnum upfrom 0
						     do
						     (setf (nth nth (backward-result ,,(car backward-args)))
							   (or (nth nth ,',result-tmp) arg)))
					       (nth 0 (backward-result ,,(car backward-args)))))))))

       ;; Note:
       ;; 1. define-static-node内でbackwardを(values nil tensor1)とする
       ;; 2. 遅延評価しながらBackwardを構築するとき、コンパイル時に上の関数の返り値がわからない
       ;; 3. とりあえず入力と同じTensorを用いてBackwardを構築する
       ;; 4. ↑ はPruneされると思うけど... 万一計算のーどが繋がっちゃった時にWarningを出す必要がある。
       ;; This is a main part of composite-node.
       
       (define-and-impl-node (,name (,self-name ,@constructor-args)
			      :device t
			      :where ,where
			      :slots (,@slots
				      ,@save-for-backward-slots
				      (forward-arguments :initform nil :accessor forward-args)
				      (backward-result :initform nil :accessor backward-result))
			      :out-scalar-p ,out-scalar-p
			      :documentation ,documentation
			      :cache-when-compiled t
			      :forward ((,@forward-args)
					(flet ((,forward-flet-name (,@forward-args)
						 (declare (ignorable ,(car forward-args)))
						 (with-composite-node-mode ,(car forward-args)
						   (locally ,@forward-body))))
					  `(progn
					     (setf (forward-args ,,(car forward-args)) (list ,,@forward-args))
					     (funcall ,#',forward-flet-name ,,@forward-args))))
			      :backward ((,@backward-args) ;; = self dout
					 ;; Initializes backward node with (Backward self)
					 (let ((bw-node (,backward-node-name ,(car backward-args))))
					   ;; First-Argument = Forward of Backward-Node
					   ;; Cdr-Th Argument = Instant-Kernel
					   (values
					    ,@(loop for nth fixnum upfrom 0
						    for input-var in (cdr forward-args) ;; forward-args = (self x y z ...)
						    if (= nth 0) ;; for first argument
						      collect `(forward bw-node ,@(cdr backward-args))
						    else
						      ;; reads the result of first call.
						      collect `(with-instant-kernel ,(second backward-args) ;; = dout
								 `(nth ,,nth (backward-result ,,(car backward-args)))))))))
	 ,@constructor-body))))

(defmacro define-op ((name
		      (self &rest constructor-args)
		      &key
			(where nil)
			(slots nil)
			(out-scalar-p nil)
			(save-for-backward-names nil)
			(forward nil)
			(backward nil)
			(extends-fw nil)
			(extends-bw nil)
			(documentation ""))
		     &body constructor-body)
  "
## [macro] define-op

Defines a differentiable AbstractNode which its definition is given by a function.

```lisp
(define-op (name (self &rest constructor-args) where slots out-scalar-p save-for-backward-names forward backward documentation extends-fw extends-bw) &body body)
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
		     :extends ,extends-fw
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
		     :extends ,extends-bw)
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


