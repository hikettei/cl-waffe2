

(in-package :cl-waffe2/vm.nodes.generic-tensor.test)

(in-suite :test-tensor)


;; Problems: Compile-Speed (add config) (there's no need)
;; (10 a) (10 10) <- a = 10 (DONE)

;; Making Add/Sub/Mul/Div Nodes
;; OpenBLAS/Common Lisp backend.
;; update: construct-forward (InputTensor UI)
;; add:    construct-backward
;; view -> view.
;; ViewNode


(defun test-simple-forward ()
  (let ((out (!add (make-tensor `(10 10))
		     (make-tensor `(10 10)))))
      (forward (build out))))

(defun test-simple-forward-with-view ()
    (let ((out (!add (!view (make-tensor `(10 1)) t `(:broadcast 10))
		     (make-tensor `(10 10)))))
      (proceed out)))

(test test-forward
  (is (test-simple-forward)))

(test forward-with-view-simple-test
  (is (test-simple-forward-with-view)))

;; Tests call-with-view, view=t, dtype=uint8
(defun test-elementwise-unroll-forward ()
  (let ((a (make-tensor `(3 3) :dtype :uint8))
	(b (make-tensor `(3 3) :dtype :uint8)))
    (dotimes (i 9)
      (setf (vref a i) 1)
      (setf (vref b i) 1))
    (let ((out (!add a b)))
      (let ((model (build out)))
	(let ((result (tensor-vec (forward model))))
	  (every #'(lambda (elm) (= elm 2)) result))))))

(defun test-elementwise-forward ()
  (let ((a (make-tensor `(100 100) :dtype :uint8))
	(b (make-tensor `(100 100) :dtype :uint8)))
    (dotimes (i 10000)
      (setf (vref a i) 1)
      (setf (vref b i) 1))
    (let ((out (!add a b)))
      (let ((model (build out)))
	(let ((result (tensor-vec (forward model))))
	  (every #'(lambda (elm) (= elm 2)) result))))))

;;(test test-call-with-view
;;  (is (test-elementwise-unroll-forward))
;;  (is (test-elementwise-forward))
;;  )


;; testing embody-input

(test flexible-insert-test
  (is (= 0 (tensor-flexible-p (make-tensor `(~ 3 3)))))
  (is (= 1 (tensor-flexible-p (make-tensor `(3 ~ 3)))))
  (is (= 2 (tensor-flexible-p (make-tensor `(3 3 ~)))))
  (is (= 0 (tensor-flexible-p (make-input `(~ 3 3) nil))))
  (is (= 1 (tensor-flexible-p (make-input `(3 ~ 3) nil))))
  (is (= 2 (tensor-flexible-p (make-input `(3 3 ~) nil)))))

