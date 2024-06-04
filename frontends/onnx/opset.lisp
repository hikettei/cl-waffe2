
(in-package :cl-waffe2.frontends/onnx)

(macrolet ((def-elwise (name version op)
	     `(defop (,name ,version)
		  ((gph inputs attrs)
		    (declare (ignore attrs))
		    (assert (= 2 (length inputs)) () "Assertion Failed: ~a expects binary ops but got ~a" ,name inputs)
		    ;; [FIXME] Broadcasting semantic compapitibility?
		    ;; https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
		    (let ((a (if (cl-waffe2/vm.generic-tensor:scalar-p (first inputs))
				 (first inputs)
				 (wf:!flexible (first inputs))))
			  (b (if (cl-waffe2/vm.generic-tensor:scalar-p (second inputs))
				 (second inputs)
				 (wf:!flexible (second inputs)))))
		      (,op a b))))))
  (def-elwise "Add" 1 wf:!add)
  (def-elwise "Sub" 1 wf:!sub)
  (def-elwise "Mul" 1 wf:!mul)
  (def-elwise "Div" 1 wf:!div))

(macrolet ((def-unary (name version op)
	     `(defop (,name ,version)
		  ((gph inputs attrs)
		    (declare (ignore attrs))
		    (,op (car inputs))))))
  (def-unary "Sqrt" 1 wf:!sqrt)
  (def-unary "Relu" 1 wf/nn:!relu))

(defop ("Gemm" 1)
    ((cls inputs attrs)
      (assert (or (= (length inputs) 2) (= (length inputs) 3))
	      ()
	      "Assertion failed. Gemm should take two or three inputs but got: ~a" inputs)

      (let ((alpha (gethash "alpha" attrs))
	    (beta  (gethash "beta" attrs))
	    (transA (gethash "transA" attrs 0))
	    (transB (gethash "trabsB" attrs 0)))

	(let* ((a (if alpha (wf:!mul (car inputs) alpha) (car inputs)))
	       (b (second inputs))
	       ;; A/B has an inifinite-rank
	       (a (wf:!flexible a))
	       (b (wf:!flexible b))
	       (a (if (= 1 transA) (wf:!t a) a))
	       (b (if (= 1 transB) (wf:!t b) b))
	       (out (wf:!matmul a b))
	       (out (if (= (length inputs) 3)
			(wf:!add out (wf:!flexible (wf:!mul beta (third inputs))))
			out)))
	  out))))

(defop ("Reduce" 1)
    ((gph inputs attrs &key (reduce #'wf:!sum))
      (let ((axis (gethash "axes" attrs 0)))
	(car
	 (multiple-value-list
	  (funcall
	   reduce
	   (car inputs)
	   :axis axis
	   :keepdims
	   (if (= (gethash "keepdims" attrs 1) 1)
	       t
	       nil)))))))

(macrolet ((defreduce (name op)
	     `(defop (,name 0)
		  ((gph inputs attrs)
		    (let ((r (get-converter "Reduce" (gp-opset-version gph))))
		      (funcall r gph inputs attrs :reduce ,op))))))
  (defreduce "ReduceMax"  #'wf:!max)
  (defreduce "ReduceMin"  #'wf:!min)
  (defreduce "ReduceSum"  #'wf:!sum)
  (defreduce "ReduceMean" #'wf:!mean)
  ;; ReduceProd
  ;; ReduceLogSumExp
  )

(defop ("Pow" 13)
    ((cls inputs attrs)
      (declare (ignore attrs))
      (wf:!expt (car inputs) (second inputs))))

(defop ("Constant" 9)
    ((cls inputs attrs)
      (let ((val
	      (or
	       (gethash "value" attrs)
	       (error "No values in Constant"))))
	(if (numberp val)
	    (make-tensor val)
	    (if (and (not (stringp val)) (arrayp val))
		(change-facet val :direction 'AbstractTensor)
		(progn
		  (warn "cl-waffe2 attempts somehow convert ~a to AbstractTensor in ConstantNode using change-facet method, but this may failed."
			val)
		  (change-facet val :direction 'AbstractTensor)))))))



