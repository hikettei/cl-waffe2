
(in-package :cl-waffe2/vm.nodes)

;; TODO: =================================
;; In accordance with refactoring of defnode, :where, add this feature: checking the number of arguments, number of outputs reading :where. (At: forward :around)
;; せっかく:whereで関数宣言してるのに安全性関連の機能が貧弱すぎる・・・
;; Add Backward tests to define-static-node
;; Add Documents
;; Update documents

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
(TODO DOCS)"
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

(TODO DOCS)
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
```


(with-reading-save4bw ((a b) (b c))
 ...)
(TODO)"
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
```

(TODO)
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

Defines a differentiable composite, instantly defines as composite-function
↑EncapsulatedNodeと同じ要領で実装できる"

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
					 (forward (,backward-node-name ,(car backward-args)) ,@(cdr backward-args))))
	 ,@constructor-body))))


#|
(define-static-node (TestModel (self)
		     :where (A[~] -> OUT[~])
		     :save-for-backward-names (x-input)
		     :forward ((self x)
			       (with-setting-save4bw ((x-input x))
				 (cl-waffe2/base-impl:proceed
				  (cl-waffe2/base-impl:!sin x))))
		     :backward ((self dout)
				(with-reading-save4bw ((x x-input))
				  (values
				   (cl-waffe2/base-impl:proceed
				    (cl-waffe2/base-impl:!mul
				     dout
				     (cl-waffe2/base-impl:!cos x))))))))
|#

