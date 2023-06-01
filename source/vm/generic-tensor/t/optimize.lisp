
(in-package :cl-waffe2/vm.nodes.generic-tensor.test)

(in-suite :test-tensor)


(defnode (1DFunc (self)
	  :where `([~] -> [~])
	  :documentation ""))

(define-impl (1DFunc :device cl-waffe2/backends.lisp:LispTensor)
	     :forward ((self x)   `(progn
				     ;;(print "1DFunc Called")
				     ,x))
	     :backward ((self dout dx) (values dout)))

(defun !f (tensor)
  (forward (1DFunc) (!copy tensor)))

(defun build-node1 ()
  (with-devices (cl-waffe2/backends.lisp:LispTensor)
    (let* ((input (make-input `(batch-size n) :input))
	   (weight (make-tensor `(100 100)))
	   (out1 (!add (!f input) weight))
	   (out2 (!add (!f out1) weight))
	   (out3 (!add (!f out2) weight)))
      (multiple-value-bind (forward bw vars params) (build out3)
	(viz-computation-node out3 "./assets/out1.dot")
	(embody-input vars :input (make-tensor `(100 100)))
	(funcall forward)))))

(defun build-node2 ()
  (with-devices (cl-waffe2/backends.lisp:LispTensor)
    (let* ((input (make-input `(batch-size n) :input))
	   (val   (make-tensor `(100 100)))
	   (weight (make-tensor `(100 100)))
	   (l (!mul input weight))
	   (k (!f l))
	   (out1 (!add k weight))
	   (out2 (!add k weight))
	   (out3 (!mul out1 out2)))
      
      (multiple-value-bind (forward bw vars params) (build out3)
	(viz-computation-node out3 "./assets/out2.dot")
	(embody-input vars :input val)
	(funcall forward)))))

(defun build-node3 ()
  (with-devices (cl-waffe2/backends.lisp:LispTensor)
    (let* ((x (make-tensor `(100 100)))
	   (y (make-tensor `(100 100)))
	   (z (!sum (!add x (!f (!f (!f (!f (!f (!f y))))))))))

      (multiple-value-bind (forward bw vars params) (build z)
	(viz-computation-node z "./assets/out3.dot")
	(funcall forward)))))

;; TODO: Compare the results with no-optim ver.
;; dot -Tpng ./assets/out.dot > ./assets/out.png
(test node-optimize-test
  (is (build-node1))
  (is (build-node2))
  (is (build-node3)))

