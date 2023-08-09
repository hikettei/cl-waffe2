
(in-package :cl-waffe2/vm)

(declaim (inline maybe-read-result write-result apply-instruction))
(declaim (ftype (function (AbstractTensor) AbstractTensor) maybe-read-result))
(defun maybe-read-result (tensor)
  (declare (type AbstractTensor tensor))
  (let* ((state (tensor-state tensor))
	 (res
	   (or (when state
		 (the AbstractTensor
		      (nth
		       (tensor-out-n tensor)
		       (cl-waffe2/vm.generic-tensor::statecontainer-forward-result state))))
	       tensor)))
    res))

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

(defun accept-instructions (iseq)
  ""
  (declare (type list iseq)
	   (optimize (speed 3)))

  (when iseq
    (loop for inst of-type WFInstruction in iseq
	  ;; TODO: Runtime Shape Inspection etc...
	  do (write-result (wfop-self inst) (apply-instruction inst))
	  finally
	     (return-from accept-instructions (maybe-read-result (wfop-self inst))))))


