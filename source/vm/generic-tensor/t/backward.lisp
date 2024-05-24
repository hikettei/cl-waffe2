
;; Test if Chain Rule is working

(in-package :cl-waffe2/vm.nodes.generic-tensor.test)

(in-suite :test-tensor)

(defun chain-test1 ()
  (let ((a (make-tensor `(15 15) :initial-element 1.0 :requires-grad t)))
    (proceed-backward (!mul (!sum a) 1.0))
    (~= (vref (grad a) 0) 1.0)))

;; !mul/!div Swapping/X->X,Y
(defun chain-test2 ()
  (let ((a (make-tensor `(15 15) :initial-element 1.0 :requires-grad t)))
    (proceed-backward (!div (!sum a) 1.0))
    (~= (vref (grad a) 0) 1.0)))

;; No side eff?
(defun chain-test3 ()
  (let ((a (make-tensor `(15 15) :initial-element 1.0 :requires-grad t)))
    (proceed-backward (!div (!sum a) 2.0))
    (~= (vref (grad a) 0) 0.5)))

(defun chain-test4 ()
  (let ((a (make-tensor `(15 15) :initial-element 1.0 :requires-grad t))
	(b (make-tensor (* 15.0 15.0) :requires-grad t)))
    ;; (!div scal mat) -> (!mul (!sas-div 1 scal) mat)
    (proceed-backward (!div (!sum a) b))
    (~= (vref (grad a) 0) (float (/ (* 15 15))))))

(defun chain-test5 ()
  (let ((a (make-tensor `(15 15) :initial-element 1.0 :requires-grad t))
	(b (* 15.0 15.0)))
    ;; (!div scal mat) -> (!mul (!sas-div 1 scal) mat)
    (proceed-backward (!div (!sum a) b))
    (~= (vref (grad a) 0) (float (/ (* 15 15))))))
;; =======================================================================

(defun chain-test6 ()
  (let ((a (make-tensor `(15 15) :initial-element 1.0 :requires-grad t))
	(b (* 15.0 15.0)))
    (proceed-backward (!sum (!add (!sin a) b)))
    (~= (vref (grad a) 0) (cos 1))))

(defun chain-test7 ()
  (let ((a (make-tensor `(15 15) :initial-element 1.0 :requires-grad t)))
    (proceed-backward (!sin (!sin a)))
    (~= (vref (grad a) 0) (* (cos (sin 1)) (cos 1)))))

(defun backward-being-not-destructed ()
  (let ((a (parameter (make-tensor `(15 15) :initial-element 3 :requires-grad t)))
	(b (parameter (make-tensor `(15 15) :initial-element 6 :requires-grad t))))
    (proceed-backward (!sum (!mul (!sin a) (!sin b))))
    (and
     (~= (vref a 0) 3.0)
     (~= (vref b 0) 6.0))))

(defun chain-test8 ()
  (let ((a (parameter (cl-waffe2/distributions:ax+b `(3 3) 0 3)))
	(b (parameter (cl-waffe2/distributions:ax+b `(3 3) 0 2)))
	(c (parameter (cl-waffe2/distributions:ax+b `(3 3) 0 1))))
    (Proceed-backward (!sum (!add (!mul a b) c)))
    ;;(with-no-grad
    ;;(let ((a (!sum (!add (!mul a b) c))))
    ;;  (with-no-grad (build a))
    ;;  (cl-waffe2/viz:viz-computation-node a "./assets/hoge.dot")))
    ;;(print (grad a))
    ;;(print (grad b))
    ;;(print (grad c))
    (and
     (~= (vref (grad a) 0) 2)
     (~= (vref (grad b) 0) 3)
     (~= (vref (grad c) 0) 1))))

(defun chain-test9 ()
  (let ((a (parameter (cl-waffe2/distributions:ax+b `(3 3) 0 3)))
	(b (parameter (cl-waffe2/distributions:ax+b `(3 3) 0 2)))
	(c (parameter (cl-waffe2/distributions:ax+b `(3 3) 0 1)))
	(a1 (parameter (cl-waffe2/distributions:ax+b `(3 3) 0 2)))
	(c1 (parameter (cl-waffe2/distributions:ax+b `(3 3) 0 1))))

    (let ((prev-layer (!add (!mul a b) c)))
      (proceed-backward (!sum (!add (!mul a1 prev-layer) c1))))

    (and
     (~= (vref (grad a)  0) 4)
     (~= (vref (grad b)  0) 6)
     (~= (vref (grad c)  0) 2)
     (~= (vref (grad a1) 0) 7)
     (~= (vref (grad c1) 0) 1))))
    
     
     

(test chain-rule-test
  (is (chain-test1))
  (is (chain-test2))
  (is (chain-test3))
  (is (chain-test4))
  (is (chain-test5))
  (is (chain-test6))
  (is (chain-test7))
  (is (chain-test8))
  (is (chain-test9)))

(test backward-side-effect-test
  (is (backward-being-not-destructed)))
  
;; save-for-backward test
