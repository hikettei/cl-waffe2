
(in-package :cl-waffe2/vm)

(declaim (inline maybe-read-result write-result apply-instruction))
(declaim (ftype (function (AbstractTensor) AbstractTensor) maybe-read-result))
(defun maybe-read-result (tensor)
  (declare (type AbstractTensor tensor))
  (let* ((state (tensor-state tensor))
	 (res
	   (or (when state
		 (nth
		  (tensor-out-n tensor)
		  (cl-waffe2/vm.generic-tensor::statecontainer-forward-result state)))
	       tensor)))
    (the AbstractTensor res)))

(declaim (ftype (function (AbstractTensor list) t) write-result))
(defun write-result (tensor result)
  (let* ((state (tensor-state tensor)))
    ;; StateContainer should exist
    (setf (cl-waffe2/vm.generic-tensor::statecontainer-forward-result state) result)))

(declaim (ftype (function (WFInstruction) list) apply-instruction))
(defun apply-instruction (instruction)
  (declare (type WFInstruction instruction)
	   (optimize (speed 3)))
  (multiple-value-list
   (apply
    (the function (wfop-op instruction))
    (map 'list #'maybe-read-result (wfop-args instruction)))))

(declaim (ftype (function (list) (or null AbstractTensor)) accept-instructions))
(defun accept-instructions (iseq)
  "
## [function] accept-instructions

```lisp
(accept-instructions iseq)
```

Evaluates generated cl-waffe2 IR sequence.

`iseq[list]` an list of `WFInstruction`
"
  (declare (type list iseq)
	   (optimize (speed 3)))

  (when iseq
    (loop for inst of-type WFInstruction in iseq
	  ;; TODO: Runtime Shape Inspection etc...
	  do (write-result (wfop-self inst) (apply-instruction inst))
	  finally
	     (return-from accept-instructions (maybe-read-result (wfop-self inst))))))


(defun make-backward-instruction (toplevel dout-mock nth leaves fuse-p)
  (let* ((dout-input (make-input (shape dout-mock) nil
				 :create-from dout-mock
				 :scalar-p (scalar-p dout-mock)
				 :dtype (dtype dout-mock)
				 :order (order dout-mock)))
	 (bw (nth nth
		  (apply
		   #'compiler-expand-backward
		   (tensor-backward toplevel)
		   dout-input
		   (tensor-variables toplevel))))
	 (iseq (reverse (node-compile-into-vm bw :fuse-p fuse-p))))
    (when (not fuse-p)
      (apply-in-place-mutation! iseq leaves))
    (setf (tensor-state dout-input)
	  (make-statecontainer :forward-out-form (make-compiled-kernel)))
    (values
     #'(lambda (dout)
	 (declare (optimize (speed 3))
		  ;; inline accept-instructions?
		  (type AbstractTensor dout))
	 (write-result dout-input (list (maybe-read-result dout)))
	 (if iseq
	     (accept-instructions iseq)
	     dout))
     #'(lambda ()
	 (format nil "Block -> ~a-BACKWARD {
~a    }
  "
		 (class-name (class-of (tensor-backward toplevel)))
		 (with-output-to-string (out)
		   (with-indent-to iseq
		     (dolist (i iseq)
		       (let ((*node-indent* (+ 4 *node-indent*)))
			 (format out "        ~a" i))))))))))



