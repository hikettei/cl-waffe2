

(in-package :cl-waffe2/base-impl.test)

;; Testing APIs provides by cl-waffe2/base-impl

;; !add !sub !mul !div
;; reshape proceed proceeed-backward !view ->scal ->mat
;; permute with proceeded result.

(defmacro lproceed (tensor)
  `(with-devices (cl-waffe2/backends.lisp:LispTensor)
     (proceed ,tensor)))

;; =============================================================
;; Testing general-purpose arithmetic APIs: !add !sub !mul !div.
;; =============================================================

(in-suite :base-impl-test)

(test test-add-form
  ;; Scalar And Scalar
  (is (= (tensor-vec
	  (lproceed (!add 1.0 1.0)))
	 2.0))
  ;; Scalar And Matrix
  (is (= (vref
	  (lproceed (!add (make-tensor `(10 10) :initial-element 1.0) 1.0))
	  0)
	 2.0))
  ;; Matrix and Scalar
  (is (= (vref
	  (lproceed (!add 1.0 (make-tensor `(10 10) :initial-element 1.0)))
	  0)
	 2.0))

  ;; Matrix and matrix
  (is (= (vref
	  (lproceed (!add (make-tensor `(10 10) :initial-element 1.0) (make-tensor `(10 10) :initial-element 1.0)))
	  0)
	 2.0)))


(test test-sub-form
  ;; Scalar And Scalar
  (is (= (tensor-vec
	  (lproceed (!sub 1.0 1.0)))
	 0.0))
  ;; Scalar And Matrix
  (is (= (vref
	  (lproceed (!sub (make-tensor `(10 10) :initial-element 2.0) 1.0))
	  0)
	 1.0))
  ;; Matrix and Scalar
  (is (= (vref
	  (lproceed (!sub 1.0 (make-tensor `(10 10) :initial-element 2.0)))
	  0)
	 -1.0))

  ;; Matrix and matrix
  (is (= (vref
	  (lproceed (!sub (make-tensor `(10 10) :initial-element 1.0)
			  (make-tensor `(10 10) :initial-element 1.0)))
	  0)
	 0.0)))


(test test-mul-form
  ;; Scalar And Scalar
  (is (= (tensor-vec
	  (lproceed (!mul 1.0 1.0)))
	 1.0))
  ;; Scalar And Matrix
  (is (= (vref
	  (lproceed (!mul (make-tensor `(10 10) :initial-element 2.0) 3.0))
	  0)
	 6.0))
  ;; Matrix and Scalar
  (is (= (vref
	  (lproceed (!mul 1.0 (make-tensor `(10 10) :initial-element 3.0)))
	  0)
	 3.0))

  ;; Matrix and matrix
  (is (= (vref
	  (lproceed (!mul (make-tensor `(10 10) :initial-element 1.0)
			  (make-tensor `(10 10) :initial-element 1.0)))
	  0)
	 1.0)))


(test test-div-form
  ;; Scalar And Scalar
  (is (= (tensor-vec
	  (lproceed (!div 1.0 1.0)))
	 1.0))
  ;; Scalar And Matrix
  (is (= (vref
	  (lproceed (!div (make-tensor `(10 10) :initial-element 6.0) 3.0))
	  0)
	 2.0))
  ;; Matrix and Scalar
  (is (= (vref
	  (lproceed (!div 10.0 (make-tensor `(10 10) :initial-element 2.0)))
	  0)
	 5.0))

  ;; Matrix and matrix
  (is (= (vref
	  (lproceed (!mul (make-tensor `(10 10) :initial-element 1.0)
			  (make-tensor `(10 10) :initial-element 1.0)))
	  0)
	 1.0)))

;; =============================================================
;; Testing !reshape ->scal ->mat !unsqueeze !squeeze
;; =============================================================

;; Note: !view ... testing with !sum !mean

(test reshaping-test
  (is (equal (shape (!reshape (make-tensor `(10 10)) 100))
	     `(100)))
  (is (equal (shape (!reshape (make-tensor `(10 10)) t))
	     `(100))))

(test ->scal-test
  (is (scalar-p (->scal (make-tensor `(1 1))))))

(test ->mat-test
  (is (equal `(1)   (shape (->mat (make-tensor 1.0)))))
  (is (equal `(1 1) (shape (->mat (make-tensor 1.0) :dims 2)))))

;; =============================================================
;; Testing proceed proceed-backward
;; =============================================================

;; =============================================================
;; Testing !flexible, broadcasting
;; =============================================================

;; TODO: It isn't enough
(test flexible-test
  (is (!add (make-tensor `(10 10)) (!flexible (make-tensor `(10))))))


;; =============================================================
;; Testing Differentiable Proceed
;; =============================================================

;; out = C + A * B
(defun proceed-continue-test (n)
  (let ((a (parameter (ax+b `(10 10) 0 3)))
	(b (parameter (ax+b `(10 10) 0 2)))
	(c (parameter (ax+b `(10 10) 0 4)))

	(a1 (parameter (ax+b `(10 10) 0 3)))
	(b1 (parameter (ax+b `(10 10) 0 2)))
	(c1 (parameter (ax+b `(10 10) 0 4))))

    (multiple-value-bind (ag bg cg)
	(normal-proceed a b c)
      (multiple-value-bind (ag1 bg1 cg1)
	  (composed-proceed a1 b1 c1)
	(let ((cp (list `(,ag ,ag1)
			`(,bg ,bg1)
			`(,cg ,cg1)))
	      (ans `(2 3 1)))
	  (and
	   (= (vref (car (nth n cp)) 0)
	      (nth n ans))
	   (= (vref (car (nth n cp)) 11)
	      (nth n ans))
	   (M= (car (nth n cp))
	       (second (nth n cp)))))))))

(defun normal-proceed (a b c)
  ;;(proceed-backward (lazy-print (!mul a b)))
  ;; SumBackward bug
  ;; Side effect bug...
  (proceed-backward (!sum (!add c (!mul a b))))
  (values (grad a) (grad b) (grad c)))

(defun composed-proceed (a b c)
  (let* ((k (proceed (!mul a b)))
 	 (out (!add k c)))
    (proceed-backward (!sum out))
    (values (grad a) (grad b) (grad c))))

(test proceed-differentiable-p
  (is (proceed-continue-test 0))
  (is (proceed-continue-test 1)) ;; 
  (is (proceed-continue-test 2)))

(test proceed-composed-test
  (is (<
       (abs
	(-
	 (tensor-vec (Proceed (->scal (!Sum (Proceed (cl-waffe2/nn::!Softmax (randn `(3 3))))))))
	 3.0))
       0.000001)))


(test reshape-permute-test
  (is (equal `(5 1 5) (shape (proceed (!t (!reshape (ax+b `(5 5) 1 0) 5 5 1))))))
  (is
   (let ((a (parameter (ax+b `(3 4 5) 1 0))))
     (proceed-backward
      (!sum (!permute (!reshape a 10 3 2 1) 0 3 2 1)))
     (grad a)))
  (is
   (let ((a (parameter (ax+b `(3 4 5) 1 0))))
     (proceed-backward
      (!sum (!permute a 1 0 2)))
     (grad a))))

(test permute-test
  (is (progn
	(let ((a (parameter (ax+b `(3 4 5) 1 0))))
	  (equal (shape (proceed (!permute a 1 0 2))) `(4 5 3))))))

(test permute-composed-test
  (is (progn
	(let ((a (parameter (ax+b `(3 4 5) 1 0))))
	  (proceed-backward
	   (!permute (!sin (!permute a (torch-order 2 1 0))) 0 1 2))
	  (grad a)))))

(test permute-composed-bw-test
  (is (every #'=
	     (let ((a (parameter (ax+b `(3 3) 1 0))))
	       (proceed-backward
		;;           v
		(!mul (!t (!add 1.0 a)) (ax+b `(3 3) 1 0)))
	       (tensor-vec (grad a)))
	     (let ((a (parameter (ax+b `(3 3) 1 0))))
	       (proceed-backward
		;;      v
		(!mul (!t a) (ax+b `(3 3) 1 0)))
	       (tensor-vec (grad a))))))


(defun log1p-test (x)
  (cl-waffe2/vm:compile-forward-and-backward (!loge (!add x 1.0))))

(test log1p-op-fusion-test
  (is (= (length (log1p-test (randn `(3 3)))) 2)))


(defun view-diff-test1 ()
  (let ((a (parameter (ax+b `(3 3) 0 1))))
    (proceed-backward
     (!add
      (!view a `(0 2) `(0 2))
      (randn `(2 2))))
    (grad a)))

(test view-diff-test
  (is (every #'=
	     (tensor-vec (view-diff-test1))
	     #(1 1 1 1 1 1 0 0 0))))

