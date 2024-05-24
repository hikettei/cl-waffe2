
(in-package :cl-waffe2/base-impl.test)

;; == Here, we provide testing framework. ========
;;
;; You can perform tests like:
;; (sum-tester LispTensor)
;; (mathematical-test-set LispTensor)
;;
;; ===============================================

;; To Add: node-test-tools
;; Testing:
;;
;; [X X X] <- <Nodes> <- Input
;;    ↑ (Check) Is it correct?

;; [1 1 1] -> <Nodes> -> Gradient
;;                           ↑ (Check) Is it correct?

;; Memo:
;; Check that operations are correctly defined and executed, node by node.
;; Composite several nodes -> testing will be done at generic-tensor/t


(macrolet ((define-arith-tester (name op result grad1 grad2)
	     `(define-tester ,name :all
		(let ((a (make-tensor `(100 100) :initial-element 10))
		      (b (make-tensor `(100 100) :initial-element 1)))
		  (let ((c (proceed (,op a b)))
			(all-p t))
		    (loop while all-p
			  for i fixnum upfrom 0 below 10000
			  do (setq all-p (= (vref c i) ,result)))
		    (when all-p
		      (let ((a (make-tensor `(100 100) :initial-element 10 :requires-grad t))
			    (b (make-tensor `(100 100) :initial-element 1 :requires-grad t)))
			(proceed-backward (,op a b))
			(let ((all-p t))
			  (loop while all-p
				for i fixnum upfrom 0 below 10000
				do (setq all-p (and (= (vref (grad a) i) ,grad1)
						    (= (vref (grad b) i) ,grad2))))
			  (if all-p
			      t
			      :backward)))))))))
  (define-arith-tester add-tester  !add  11 1  1)
  (define-arith-tester sub-tester  !sub  9  1 -1)
  (define-arith-tester mul-tester  !mul  10 1 10)
  (define-arith-tester div-tester  !div  10 1 -10)
  (define-arith-tester move-tester !move 1 0 1))


(macrolet ((define-scalar-mat-tester (name op result grad1 grad2)
	     `(define-tester ,name :all
		(let ((a (make-tensor `(100 100) :initial-element 10))
		      (k (make-tensor 1)))
		  (let ((c (proceed (,op k a)))
			(all-p t))
		    (loop while all-p
			  for i upfrom 0 below 10000
			  do (setq all-p (= (vref c i) ,result)))
		    (when all-p
		      (let ((a (make-tensor `(100 100) :initial-element 10 :requires-grad t))
			    (k (make-tensor 1 :requires-grad t)))
			(proceed-backward (,op k a))
			(let ((all-p t))
			  (loop while all-p
				for i upfrom 0 below 10000
				do (setq all-p (and (= (vref (grad a) i) ,grad1)
						    (= (tensor-vec (grad k)) ,grad2))))
			  (if all-p
			      t
			      :backward)))))))))
  ;; name function-name result grad-mat grad-scal
  (define-scalar-mat-tester scalar-add-tester !scalar-add 11 1 1)
  (define-scalar-mat-tester scalar-sub-tester !scalar-sub 9  1 -1)
  (define-scalar-mat-tester scalar-mul-tester !scalar-mul 10 1 10)
  (define-scalar-mat-tester scalar-div-tester !scalar-div 10 1 -10))

;; Add: Tests on
;; !matmul/!dot (<=> sum)
;; !transposed matmul
;; argmax/argmin/max/min
;; einsum
;;

(macrolet ((define-ss-tester (name op lisp-op grad1 grad2)
	     `(define-tester ,name :all
		(let ((a (make-tensor 1 :requires-grad t))
		      (b (make-tensor 1 :requires-grad t)))
		  (let ((c (proceed (,op a b)))
			(r (,lisp-op 1 1)))
		    (when (= r (tensor-vec c))
		      (proceed-backward (,op a b))
		      (if (and (= (tensor-vec (grad a)) ,grad1)
			       (= (tensor-vec (grad b)) ,grad2))
			  t
			  :backward)))))))
  (define-ss-tester ss-add-tester !add + 1 1)
  (define-ss-tester ss-sub-tester !sub - 1 -1)
  (define-ss-tester ss-mul-tester !mul * 1 1)
  (define-ss-tester ss-div-tester !div / 1 -1))


;; A.grad = (!matmul dout db.t)
;; B.grad = (!matmul da.t dout)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun matmul-dx (dout y)
    (proceed (!matmul dout (!t y))))

  (defun matmul-dy (dout x)
    (proceed (!matmul (!t x) dout))))

(define-tester matmul-tester :dense
  (let* ((a (ax+b `(3 3) 1 0 :order :column))
 	 (b (ax+b `(3 3) 1 0 :order :column))
	 (result (proceed (!matmul a b))))
    (when (every #'= (tensor-vec result) #(15.0 18.0 21.0 42.0 54.0 66.0 69.0 90.0 111.0))
      t)))

(define-tester transpose-matmul-tester :dense
  (let* ((a (ax+b `(3 3) 1 0 :order :column))
 	 (b (ax+b `(3 3) 1 0 :order :column))
	 (result (proceed (!matmul a b))))
    (when (every #'= (tensor-vec result) #(15.0 18.0 21.0 42.0 54.0 66.0 69.0 90.0 111.0))
      t)))

(define-tester matmul-tester-mnk :dense
  (let* ((a (ax+b `(3 4) 1 0 :order :column))
	 (b (ax+b `(3 4) 1 0 :order :column)))
    (with-no-grad
      (every #'~= (tensor-vec (proceed (!matmul a (!t b))))
	     #(14.0 38.0 62.0 38.0 126.0 214.0 62.0 214.0 366.0)))))

(define-tester matmul-tester-mnk1 :dense
  (let* ((a (ax+b `(3 4) 1 0 :order :column))
	 (b (ax+b `(3 4) 1 0 :order :column)))
    (with-no-grad
      (every #'~= (tensor-vec (proceed (!matmul (!t a) b)))
	     #(80.0 92.0 104.0 116.0 92.0 107.0 122.0 137.0 104.0 122.0 140.0 158.0 116.0 137.0 158.0 179.0)))))

(define-tester matmul-both-transposed :dense
  (let* ((a (ax+b `(3 3) 1 0 :order :column))
 	 (b (ax+b `(3 3) 1 0 :order :column))
	 (result (proceed (!matmul (!t a) (!t b)))))
    (when (every #'= (tensor-vec result) #(15.0 42.0 69.0 18.0 54.0 90.0 21.0 66.0 111.0))
      t)))

(define-tester matmul-backward-test-square-sparse :dense
  (let ((a (parameter (ax+b `(3 3) 1 0 :order :column)))
	(b (parameter (ax+b `(3 3) 2 0 :order :column)))
	(dout         (ax+b `(3 3) 0 1 :order :column)))
    ;; A @ B
    (proceed-backward (!matmul a b))
    (and
     (M= (grad a) (matmul-dx dout b))
     (M= (grad b) (matmul-dy dout a)))))

(define-tester matmul-backward-test-square-dense :dense
  (let ((a (parameter (randn `(3 3) :order :column)))
	(b (parameter (randn `(3 3) :order :column)))
	(dout         (ax+b `(3 3) 0 1 :order :column)))
    (proceed-backward (!matmul a b))
    (and
     (M= (grad a) (matmul-dx dout b))
     (M= (grad b) (matmul-dy dout a)))))

(defun matmul-3x4-4x3-test ()
  (with-devices (cl-waffe2/backends.cpu:CPUTensor cl-waffe2/backends.lisp:LispTensor)
    (let ((a (parameter (randn `(3 4) :order :column)))
	  (b (parameter (randn `(4 3) :order :column)))
	  (dout         (ax+b `(3 3) 0 1 :order :column)))
      (proceed-backward (!matmul a b))
      (and
       (M= (grad a) (matmul-dx dout b))
       (M= (grad b) (matmul-dy dout a))))))

(deftest matmul-3x4-4x3-test
  (ok (matmul-3x4-4x3-test)))

;; Continue...
(macrolet ((define-matmul-test-form (name matmul-form size1 size2)
	     `(define-tester ,name :dense
		(let* ((a (parameter (randn ,size1 :order :column)))
		       (b (parameter (randn ,size2 :order :column)))
		       (node-out ,matmul-form)
		       (dout (ax+b (shape node-out) 0 1 :order :column)))
		  (proceed-backward node-out)
		  (and
		   (M= (grad a) (matmul-dx dout b))
		   (M= (grad b) (matmul-dy dout a)))))))
  (define-matmul-test-form
      matmul-test-form-test
      (!matmul a b)
    `(3 3)
    `(3 3))

  )
		

;; should be:
;; 9 12 15
;; 9 12 15
;; 9 12 15 
;; 9 12 15

(deftest adding-gradients-with-permution-shuffled-test
  (ok (let ((a (parameter (randn `(4 3)))))
	(proceed-backward (!matmul (ax+b `(3 3) 1 0) (!t a)))
        (every #'= (tensor-vec (grad a))
	       #(9.0 12.0 15.0 9.0 12.0 15.0 9.0 12.0 15.0 9.0 12.0 15.0)))))

;;(eval-when (:compile-toplevel :load-toplevel :execute)
(export 'matmul-test-set)
(defmacro matmul-test-set (backend)
  `(progn;;eval-when (:compile-toplevel :load-toplevel :execute)
     (matmul-tester ,backend)
     (transpose-matmul-tester ,backend)
     (matmul-tester-mnk ,backend)
     (matmul-tester-mnk1 ,backend)
     (matmul-both-transposed ,backend)

     (matmul-test-form-test ,backend)
     (matmul-backward-test-square-sparse ,backend)
     (matmul-backward-test-square-dense ,backend)

     
     ))

(ss-add-tester ScalarTensor)
(ss-sub-tester ScalarTensor)
(ss-mul-tester ScalarTensor)
(ss-div-tester ScalarTensor)

(define-tester max-tester :all
  (let ((a (ax+b `(9 9) 1 0)))
    (let ((out (tensor-vec (proceed (!max a)))))
      (every #'= out
	     #(8 17 26 35 44 53 62 71 80)))))

(define-tester min-tester :all
  (let ((a (ax+b `(9 9) 1 0)))
    (let ((out (tensor-vec (proceed (!min a)))))
      (every #'= out
	     #(0 9 18 27 36 45 54 63 72)))))


;; Comparison Test

(define-tester lt-tester :all
  (let ((a (ax+b `(9 9) 0 1))
	(b (ax+b `(9 9) 0 2)))
    (let ((out (proceed (A<B a b))))
      (every #'(lambda (x) (= x 1)) (tensor-vec out)))))

(define-tester lt-tester1 :all
  (let ((a (ax+b `(9 9) 0 2))
	(b (ax+b `(9 9) 0 1)))
    (let ((out (proceed (A<B a b))))
      (every #'(lambda (x) (= x 0)) (tensor-vec out)))))

(define-tester le-tester :all
  (let ((a (ax+b `(9 9) 0 1))
	(b (ax+b `(9 9) 0 1)))
    (let ((out (proceed (A<=B a b))))
      (every #'(lambda (x) (= x 1)) (tensor-vec out)))))

(define-tester le-tester1 :all
  (let ((a (ax+b `(9 9) 0 2))
	(b (ax+b `(9 9) 0 1)))
    (let ((out (proceed (A<=B a b))))
      (every #'(lambda (x) (= x 0)) (tensor-vec out)))))



(define-tester gt-tester :all
  (let ((a (ax+b `(9 9) 0 1))
	(b (ax+b `(9 9) 0 2)))
    (let ((out (proceed (A>B a b))))
      (every #'(lambda (x) (= x 0)) (tensor-vec out)))))

(define-tester gt-tester1 :all
  (let ((a (ax+b `(9 9) 0 2))
	(b (ax+b `(9 9) 0 1)))
    (let ((out (proceed (A>B a b))))
      (every #'(lambda (x) (= x 1)) (tensor-vec out)))))

(define-tester ge-tester :all
  (let ((a (ax+b `(9 9) 0 1))
	(b (ax+b `(9 9) 0 1)))
    (let ((out (proceed (A>=B a b))))
      (every #'(lambda (x) (= x 1)) (tensor-vec out)))))

(define-tester ge-tester1 :all
  (let ((a (ax+b `(9 9) 0 2))
	(b (ax+b `(9 9) 0 1)))
    (let ((out (proceed (A>=B a b))))
      (every #'(lambda (x) (= x 1)) (tensor-vec out)))))

(define-tester eq-tester :all
  (let ((a (ax+b `(9 9) 0 1))
	(b (ax+b `(9 9) 0 1)))
    (let ((out (proceed (A=B a b))))
      (every #'(lambda (x) (= x 1)) (tensor-vec out)))))



;; scal


(define-tester slt-tester :all
  (let ((a (ax+b `(9 9) 0 1)))
    (let ((out (proceed (A<scal a 2))))
      (every #'(lambda (x) (= x 1)) (tensor-vec out)))))

(define-tester slt-tester1 :all
  (let ((a (ax+b `(9 9) 0 2)))
    (let ((out (proceed (A<scal a 1))))
      (every #'(lambda (x) (= x 0)) (tensor-vec out)))))

(define-tester sle-tester :all
  (let ((a (ax+b `(9 9) 0 1)))
    (let ((out (proceed (A<=scal a 1))))
      (every #'(lambda (x) (= x 1)) (tensor-vec out)))))

(define-tester sle-tester1 :all
  (let ((a (ax+b `(9 9) 0 2)))
    (let ((out (proceed (A<=scal a 1))))
      (every #'(lambda (x) (= x 0)) (tensor-vec out)))))



(define-tester sgt-tester :all
  (let ((a (ax+b `(9 9) 0 1)))
    (let ((out (proceed (A>scal a 2))))
      (every #'(lambda (x) (= x 0)) (tensor-vec out)))))

(define-tester sgt-tester1 :all
  (let ((a (ax+b `(9 9) 0 2)))
    (let ((out (proceed (A>scal a 1))))
      (every #'(lambda (x) (= x 1)) (tensor-vec out)))))

(define-tester sge-tester :all
  (let ((a (ax+b `(9 9) 0 1)))
    (let ((out (proceed (A>=scal a 1))))
      (every #'(lambda (x) (= x 1)) (tensor-vec out)))))

(define-tester sge-tester1 :all
  (let ((a (ax+b `(9 9) 0 2)))
    (let ((out (proceed (A>=scal a 1))))
      (every #'(lambda (x) (= x 1)) (tensor-vec out)))))

(define-tester seq-tester :all
  (let ((a (ax+b `(9 9) 0 1)))
    (let ((out (proceed (A=scal a 1))))
      (every #'(lambda (x) (= x 1.0)) (tensor-vec out)))))



(export 'comparison-test-set)

(defmacro comparison-test-set (backend)
  `(progn
     (lt-tester ,backend)
     (lt-tester1 ,backend)
     (le-tester ,backend)
     (le-tester1 ,backend)

     (gt-tester ,backend)
     (gt-tester1 ,backend)
     (ge-tester ,backend)
     (ge-tester1 ,backend)

     (eq-tester ,backend)

     (slt-tester ,backend)
     (slt-tester1 ,backend)
     (sle-tester ,backend)
     (sle-tester1 ,backend)

     (sgt-tester ,backend)
     (sgt-tester1 ,backend)
     (sge-tester ,backend)
     (sge-tester1 ,backend)
     
     (seq-tester ,backend)

     ))

(add-tester LispTensor)
(sub-tester LispTensor)
(mul-tester LispTensor)
(div-tester LispTensor)
(move-tester LispTensor)
(matmul-test-set LispTensor)
(scalar-add-tester LispTensor)
(scalar-sub-tester LispTensor)
(scalar-mul-tester LispTensor)
(scalar-div-tester LispTensor)
(comparison-test-set LispTensor)

(max-tester LispTensor)
(min-tester LispTensor)

