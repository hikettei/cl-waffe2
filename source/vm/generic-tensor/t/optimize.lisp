
(in-package :cl-waffe2/vm.nodes.generic-tensor.test)

;; [Deprecated]

(defnode (1DFunc (self)
	  :where (A[~] -> A[~])
	  :documentation ""))

(define-impl (1DFunc :device cl-waffe2/backends.lisp:LispTensor)
	     :forward ((self x)   `(progn
				     ;;(print "1DFunc Called")
				     ,x))
	     :backward ((self dout dx) (values dout)))

(defun !f (tensor)
  (forward (1DFunc) (!copy tensor)))

(defun build-node1 ()
  (let* ((input (make-input `(batch-size n) :input))
	 (weight (make-tensor `(100 100)))
	 (out1 (!add (!f input) weight))
	 (out2 (!add (!f out1) weight))
	 (out3 (!add (!f out2) weight)))
    (let ((model (build out3)))
      ;;(viz-computation-node out3 "./assets/out1.dot")
      ;;(embody-input vars :input (make-tensor `(100 100)))
      (forward model))))

(defun build-node2 ()
  (let* ((input (make-input `(batch-size n) :input))
	 (val   (make-tensor `(100 100)))
	 (weight (make-tensor `(100 100)))
	 (l (!mul input weight))
	 (k (!f l))
	 (out1 (!add k weight))
	 (out2 (!add k weight))
	 (out3 (!mul out1 out2)))
    
    (let ((model (build out3)))
      ;;(viz-computation-node out3 "./assets/out2.dot")
      ;;(embody-input vars :input val)
      (forward model))))

(defun build-node3 ()
  (let* ((x (make-tensor `(100 100)))
	 (y (make-tensor `(100 100)))
	 (z (!sum (!add x (!f (!f (!f (!f (!f (!f y))))))))))

    (let ((model (build z)))
      ;;(viz-computation-node z "./assets/out3.dot")
      (forward model))))

;; TODO: Compare the results with no-optim ver.
;; dot -Tpng ./assets/out.dot > ./assets/out.png
#|
(test node-optimize-test
  (is (build-node1))
  (is (build-node2))
  (is (build-node3))
  )

|#
