
;; Test if Chain Rule is working

(in-package :cl-waffe2/vm.nodes.generic-tensor.test)

(in-suite :test-tensor)

(defun chain-test1 ()
  (with-devices (cl-waffe2/backends.lisp:LispTensor)
    (let ((a (make-tensor `(15 15) :initial-element 1.0 :requires-grad t)))
      (proceed-backward (!mul (!sum a) 1.0))
      (= (vref (grad a) 0) 1.0))))

;; !mul/!div Swapping/X->X,Y
(defun chain-test2 ()
  (with-devices (cl-waffe2/backends.lisp:LispTensor)
    (let ((a (make-tensor `(15 15) :initial-element 1.0 :requires-grad t)))
      (proceed-backward (!div (!sum a) 1.0))
      (= (vref (grad a) 0) 1.0))))

;; No side eff?
(defun chain-test3 ()
  (with-devices (cl-waffe2/backends.lisp:LispTensor)
    (let ((a (make-tensor `(15 15) :initial-element 1.0 :requires-grad t)))
      (proceed-backward (!div (!sum a) 2.0))
      (= (vref (grad a) 0) 0.5))))

(defun chain-test4 ()
  (with-devices (cl-waffe2/backends.lisp:LispTensor)
    (let ((a (make-tensor `(15 15) :initial-element 1.0 :requires-grad t))
	  (b (make-tensor (* 15.0 15.0) :requires-grad t)))
      ;; (!div scal mat) -> (!mul (!sas-div 1 scal) mat)
      (proceed-backward (!div (!sum a) b))
      (= (vref (grad a) 0) (float (/ (* 15 15)))))))

(defun chain-test5 ()
  (with-devices (cl-waffe2/backends.lisp:LispTensor)
    (let ((a (make-tensor `(15 15) :initial-element 1.0 :requires-grad t))
	  (b (* 15.0 15.0)))
      ;; (!div scal mat) -> (!mul (!sas-div 1 scal) mat)
      (proceed-backward (!div (!sum a) b))
      (= (vref (grad a) 0) (float (/ (* 15 15)))))))
;; =======================================================================

;; ???
(defun chain-test6 ()
  (with-devices (cl-waffe2/backends.lisp:LispTensor)
    (let ((a (make-tensor `(15 15) :initial-element 1.0 :requires-grad t))
	  (b (* 15.0 15.0)))
      (proceed-backward (!sum (!add (!sin a) b)))
      (= (vref (grad a) 0) (cos 1)))))

(defun chain-test7 ()
  (with-devices (cl-waffe2/backends.lisp:LispTensor)
    (let ((a (make-tensor `(15 15) :initial-element 1.0 :requires-grad t)))
      (proceed-backward (!sin (!sin a)))
      (= (vref (grad a) 0) (* (cos (sin 1)) (cos 1))))))

(defun backward-being-not-destructed ()
  (with-devices (cl-waffe2/backends.lisp:LispTensor)
    (let ((a (parameter (make-tensor `(15 15) :initial-element 3 :requires-grad t)))
	  (b (parameter (make-tensor `(15 15) :initial-element 6 :requires-grad t))))
      (proceed-backward (!sum (!mul (!sin a) (!sin b))))
      (and
       (= (vref a 0) 3.0)
       (= (vref b 0) 6.0)))))

(test chain-rule-test
  (is (chain-test1))
  (is (chain-test2))
  (is (chain-test3))
  (is (chain-test4))
  (is (chain-test5))
  (is (chain-test7))
  (is (chain-test6)))

(test backward-side-effect-test
  (is (backward-being-not-destructed)))
  
;; save-for-backward test
