
(in-package :cl-waffe2/vm.generic-tensor)

;; Workflows of cl-waffe2 compiling:
;;
;; == [Interperter Mode (for proceed)] ===========================
;;
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;
;; [Global Compiled-Function Cache] <- Store Inlined functions
;;  MM 3Dx3D View=T View=T ...  -> Lambda ...
;;  MM 2Dx3D View=T View=T ...  -> Lambda ...
;; Move 3Dx3D View=T View=T ... -> Lambda ...
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; [JIT Compiler Global Cache] <- Storing JIT Compiled functions (specialized on element-wise ops)
;;  a*b+c            -> Lambda ...
;;  mean(x) + sum(x) -> Lambda...
;;
;;
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;
;; out = Sum(Matmul(Copy(A), Copy(B)))
;;                   |
;;  [In-place mutation, fusion is applied...]
;;                   |
;;                   V
;;    <Interpreter> or <Build>
;;
;; If [interpreter] is chosen, each time we call operations, we look up [Global Compiled-Function-Cache] and [JIT Compiler Global Cache] each time.
;;
;; If [build] is chosen, on the other hand, the looking up time is inlined.
;;
;;

;; run-node! is a proceed dedicated interpreter

(defparameter *force-use-build* nil "This parameter is for debug")
;; No Make-input
;; Interpreter Mode
(defun run-node! (toplevel
		  &key
		    (stop-me nil)
		    (called-with-vars nil)
		    (compile-option `(optimize (speed 3))))
  "
## [function] run-node!
"
  (declare (type AbstractTensor toplevel)
	   (type boolean stop-me)
	   (optimize (speed 3)))

  (when (or stop-me
	    (null (tensor-state toplevel)))
    (return-from run-node! toplevel))

  (when (detach-p toplevel)
    (setq stop-me t))
  
  
  (let* ((state (tensor-state toplevel))
	 (vars  (tensor-variables toplevel))
	 (node  (tensor-backward toplevel))
	 (compiled-fw (statecontainer-forward-out-form state)))

    (let ((next-states (map 'list #'(lambda (x) (run-node! x :stop-me stop-me :called-with-vars toplevel :compile-option compile-option)) vars)))
      (register-variables vars)

      (when (or (null (statecontainer-forward-result state))
		(and *calling-backward-mode* (car (statecontainer-backward-result state))))
	
	(when (and *calling-backward-mode*
		   (not (car (statecontainer-backward-result state))))
	  (setf (statecontainer-backward-result state)
		(list (null (statecontainer-forward-result state)))))

	(setf (statecontainer-forward-result state)
	      (multiple-value-list (apply #'funcall-cached-function compiled-fw compile-option next-states))))
      
      ;; Calling User-defined JIT Compiler

      ;; = [FIXME] ======
      ;; JIT isn't working on interpreter mode.
      ;; eval?
      
      
      (when node
	(let ((result (cl-waffe2/vm.nodes:on-finalizing-compiling
		       node
		       toplevel
		       called-with-vars
		       t)))
	  (when result
	    (eval result))))
      

      ;; on-calling-finalizing...
      (nth (tensor-out-n toplevel) (statecontainer-forward-result state)))))

;; Bottleneck:
;; A ton of creation of make-input (forward would be done without creation.)
;; init-optimizer-utils! -> when called with interpret-mode, replace them with proceed.
(defun run-node-backward! (toplevel past-dy &key (compile-option `(optimize (speed 3))) (seen nil))
  (declare (type AbstractTensor toplevel past-dy)
	   (type list seen)
	   (optimize (speed 3)))

  (when (find (tensor-iid toplevel) seen :test #'eq)
    (return-from run-node-backward!))
  
  (push (tensor-iid toplevel) seen)
  ;; Adding Gradients
  (when (null (tensor-backward toplevel))
    (return-from run-node-backward!
      (when (slot-value toplevel 'requires-grad)
	;; TODO: Later, Replace it by interpreter proceed
	(if (scalar-p toplevel)
	    (locally (declare (optimize (speed 1)))
	      (incf (tensor-vec (grad toplevel)) (tensor-vec past-dy)))
	    (with-no-grad
	      (detach! past-dy)
	      (if (= (the fixnum (tensor-grad-count toplevel)) 0)
		  (run-node!
		   (cl-waffe2/vm.nodes:forward
		    (cl-waffe2/base-impl:MoveTensorNode (dtype toplevel) :save-for-backward t)
		    (grad toplevel)
		    past-dy))
		  (run-node!
		   (cl-waffe2/vm.nodes:forward
		    (cl-waffe2/base-impl:AddNode (dtype toplevel))
		    (grad toplevel)
		    past-dy)))
	      (setf (detach-p past-dy) nil))))))

  (cl-waffe2/vm.nodes:with-shape-checkpoint (:backward (tensor-backward toplevel))
    (let* ((outs (apply
		  ;; (backward self dout dx dy dz ...)
		  ;; -> (backward self dout)
		  #'cl-waffe2/vm.nodes:expand-backward-instant
		  ;; Here, we trace the definition of backward.
		  (tensor-backward toplevel)
		  past-dy
		  compile-option
		  (tensor-variables toplevel))))

      (let ((gradients (loop for g in outs
			     for var in (tensor-variables toplevel)
			     if (and g (ancestor-param-p var))
			       collect (cl-waffe2/vm.nodes:call-instant-backward g)
			     else
			       collect nil)))
	(loop for var    in (tensor-variables toplevel)
	      for kernel in outs
	      for grad   in gradients
	      if (and kernel
		      (ancestor-param-p var))
		do (run-node-backward! var grad :compile-option compile-option :seen seen))))))

(defun vm-forward-function (toplevel
			    &key
			      (compile-mode :default))
  "
## [function] vm-forward-function
"

  (declare (type compile-option-t compile-mode))
  ;; Pruning unused nodes.
  (optimize-computation-node! toplevel :speed 1)

  #'(lambda (compiled-composite)
      (declare (type Compiled-Composite compiled-composite))
      (let* ((*node-parameters-tmp*))
	(prog1
	    (run-node! toplevel :compile-option (compile-option-form compile-mode))
	  (setf (slot-value compiled-composite 'variables) (construct-variables-table *node-parameters-tmp*))))))

(defun vm-backward-function (toplevel
			     &key
			       (compile-mode :default))
  "
## [function] vm-backward-function
"
  (declare (type compile-option-t compile-mode))

  (let* ((dout-toplevel (if (scalar-p toplevel)
			    (make-tensor 1
					 :dtype (dtype toplevel)
					 :order (order toplevel))
			    (make-tensor (shape toplevel)
					 :dtype (dtype toplevel)
					 :order (order toplevel)
					 :initial-element 1)))
	 (backward-caller (lambda ()
			    (let ((*calling-backward-mode* t))
			      (run-node-backward! toplevel dout-toplevel :compile-option (compile-option-form compile-mode))
			      t))))
    backward-caller))


(defun vm-build (toplevel
		 &key
		   (construct-backward? (not *no-grad*))
		   (compile-mode :fastest))
  "
## [function] vm-build

```lisp
(vm-build toplevel
	      &key
		(construct-backward? (not *no-grad*))
		(compile-mode :fastest))
```
"
  (declare (type AbstractTensor toplevel))

  ;;(reset-compiled-function-cache!) 

  (when *force-use-build*
    (return-from vm-build
      (build toplevel :construct-backward? construct-backward? :compile-mode compile-mode)))

  (let* ((forward (vm-forward-function toplevel :compile-mode compile-mode))
	 (backward (when construct-backward? (vm-backward-function toplevel :compile-mode compile-mode)))
	 (model (make-instance 'Compiled-Composite
			       :variables (construct-variables-table nil)
			       :compiled-forward  forward
			       :compiled-backward backward)))

    (setf (slot-value model 'compiled-forward) #'(lambda () (funcall forward model)))
    model))

