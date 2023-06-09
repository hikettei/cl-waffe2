
(in-package :cl-waffe2/distributions)

;; = References ===============================================
;; https://andantesoft.hatenablog.com/entry/2023/04/30/183032
;; Marsaglia, G., & Tsang, W. W. (2000). The ziggurat method for generating random variables. Journal of statistical software.
;; https://marui.hatenablog.com/entry/2023/01/23/194507
;; ============================================================

;; TODO: Optimize
;; TODO: Embedding Constant Table Into the code.
(defparameter *table-size* 256)
;; Generates a table.
;; TODD: Implement Modified Ziggurat

(defun ~= (x y)
  (declare (type number x y))
  (< (abs (- x y)) (typecase x
		     (single-float 0.0001)
		     (double-float 0.0000001)
		     (T 0.001))))

;; No Need To Optimize this function
(define-with-typevar-dense (ziggurat-make-table u)
    (pdf ;; Proability Density Function (Lambda)
     cdf ;; Cumulative Distribution Function
     ipd ;; Inversed CDF
     &key
     (table-size *table-size*)) ;; 2^n
  (declare ;;(optimize (safety 0)) ;; safety=0
	   (type function pdf cdf ipd)
	   (type fixnum table-size))

  (let* ((zero (coerce 0 (quote u)))
	 (+inf (typecase zero
		 (single-float most-positive-single-float)
		 (double-float most-positive-double-float)))
	 (-inf (typecase zero
		 (single-float most-negative-single-float)
		 (double-float most-negative-double-float)))
	 (x0)
	 (x0-area)
	 (rectangle-X (make-array table-size :element-type (quote u)))
	 (rectangle-Y (make-array table-size :element-type (quote u)))
	 (irr (make-array table-size :element-type (quote u))))
    (declare (type (simple-array u (*)) rectangle-X rectangle-Y irr))
    (labels ((area-of (x)
	       (declare (type u x))
	       ;; max(CDF(x)) ~= CDF(+inf)
	       ;; PDF(x)x + CDF(+inf) - CDF(x)
	       ;; Fix: ugly coerce
	       (+ (* x (coerce (funcall pdf x) (quote u)))
		  (coerce (funcall cdf +inf) (quote u))
		  (- (coerce (funcall cdf x) (quote u)))))
	     (error-of (x)
	       (declare (type u x))
	       (let ((area (area-of x))
		     (xi x))
		 ;; Area -> 0
		 
		 (loop for i fixnum upfrom 1 below table-size
		       do (progn
			    (setq xi (coerce (funcall ipd (+ (/ area xi) (funcall pdf xi))) (quote u)))
			    ;; Assertion, pdf >= 0
			    (when (< xi zero)
			      (return-from error-of -inf))))
		 ;; Check IsNAN? this is processing-system-dependant...
		 xi)))

      ;; Compute x0
      (let ((left  (coerce 2   (quote u)))
	    (right (coerce 10  (quote u)))
	    (mid   (coerce 6   (quote u)))
	    (half  (coerce 0.5 (quote u)))
	    (bit 1e-6)) ;; mid = 1/2(left+right)
	(declare (type u left right mid half))
	(loop named btree
	      while t
	      do (progn
		   (setq mid (* half (+ left right)))
		   (cond
		     ((and (~= left mid)
			   (< mid right))
		      (setq mid (+ left bit)))
		     ((and (~= right mid)
			   (< left mid))
		      (setq mid (- right bit))))
		   
		   (when (or (~= left mid)
			     (~= right mid))
		     (return-from btree))
		   (print left) (print mid) (print right)
		   (let ((left-error  (error-of left))
			 (mid-error   (error-of mid))
			 (right-error (error-of right)))
		     (print left-error)
		     (print mid-error)
		     (print right-error)
		     (cond
		       ((not (= (signum left-error)
				(signum mid-error)))
			(setq right mid))
		       ((not (= (signum right-error)
				(signum mid-error)))
			(setq left mid))
		       (T
			(error "Something went wrong when making ziggurat-table!"))))))
	(setq x0 right)
	(setq x0-area (area-of x0)))

      (setf (aref rectangle-X 0) (/ x0-area (the u (funcall pdf x0))))
      (setf (aref rectangle-X 1) x0)

      (loop for i fixnum upfrom 2 below table-size
	    do (let ((xi (the u
			      (funcall ipd
				       (the u
					    (+ (/ x0-area (aref rectangle-X (1- i)))
					       (the u (funcall pdf (aref rectangle-X (1- i))))))))))
		 (setf (aref rectangle-x i) xi)
		 (assert (> xi zero)
			 nil
			 "Asertion Failed with xi > 0, xi=~a, Parameters could be wrong..."
			 xi)))

      (loop for i fixnum upfrom 0 below table-size
	    if (= (1+ i) table-size)
	      do (setf (aref rectangle-Y i) (the u (funcall pdf zero)))
	    else
	      do (setf (aref rectangle-Y i) (the u (funcall pdf (aref rectangle-X (1+ i))))))

      (loop for i fixnum upfrom 0 below table-size
	    if (= (1+ i) table-size)
	      do (setf (aref irr i) zero)
	    else
	      do (setf (aref irr i) (/ (aref rectangle-X i)
				       (aref rectangle-X (1+ i)))))
      
      (values rectangle-X
	      rectangle-Y
	      irr))))

;; Optimize It First. Parallelize
(define-with-typevar-dense (make-ziggurat-sampler u) (rx ry ir table-size)
  (declare ;;(optimize (speed 3))
	   (type (simple-array u (*)) rx ry ir)
	   (type fixnum))
  (loop named sampling
	while t
	do (let* ((zero (coerce 0 (quote u)))
		  (u    (random most-positive-fixnum))
		  (u1   (- u 1)) ;; u1f = [-1.0, 1.0)
		  (u1f (coerce
			(/ (ash u1 -10)
			   (ash 1 53))
			(quote u)))
		  (i (random (1- table-size))))
	     (declare (type fixnum u1)
		      (type u u1f))

	     (when (< (abs u1f)
		      (aref ir i))
	       (return-from sampling (the u (* u1f (aref rx i)))))

	     (if (= i 0)
		 (loop with sp u = zero
		       with tp u = zero
		       with x0 u = (aref rx 1)
		       while (> (* sp sp) (+ tp tp))
		       do (and
			   (setq sp (-
				     (/
				      (log (- 1
					      (/
					       (ash (random most-positive-fixnum) -11)
					       (ash 1 53)))
					   x0))))
			   (setq tp (-
				     (/
				      (log (- 1
					      (/
					       (ash (random most-positive-fixnum) -11)
					       (ash 1 53)))
					   x0)))))
		       finally
			  (return-from sampling
			    (* (+ x0 sp)
			       (if (< u1 0) -1 1))))
		 (let ((x (* u1f (aref rx i)))
		       (y (coerce (/ (ash (random most-positive-fixnum) -11)
				     (ash 1 53))
				  (quote u))))
		   (when (<=
			  (+ (* y (-
				   (aref ry i)
				   (aref ry (1- i))))
			     (aref ry (1- i)))
			  (exp (* -0.5 x x)))
		     (return-from sampling x)))))))

(defun erf1 (x &aux (inf 10000.0)) ;; x -> 6.5d0, erf(x) -> 1.0d0
  "Error function."
  (if (or (> x inf) (< x (- inf)) (= x 0)) ;; If x is enough large, treated as +inf
      (signum x)
      (let ((erf (/ 2 (sqrt pi)))
	    (sum 0.0)
	    (fct 1.0))
	(loop named itr
	      while t
	      for i fixnum upfrom 0
	      do (let ((elm (*
			     (if (or (= i 0) (= i 1))
				 1
				 -1)
			     (expt x (+ (* 2 i) 1))
			     (/ (* fct (+ (* 2 i) 1))))))
		   (when (= (+ elm sum) sum)
		     (return-from itr))
		   (incf sum elm)
		   (setq fct (* fct (+ i 1)))))
	(* sum erf))))

(defun erf2 (x &aux (inf 10000.0))
  (if (or (> x inf) (< x (- inf)) (= x 0)) ;; If x is enough large, treated as +inf
      (signum x)
      (let* ((sum 0.0)
             (factorial 1.0)
             (i 0))
        (loop while t do
          (let* ((element (/ (* (if (oddp i) -1 1)
                                (expt x (+ (* 2 i) 1)))
                             (* factorial (+ (* 2 i) 1)))))
            (if (= (+ sum element) sum)
                (return sum)
                (progn
                  (setq sum (+ sum element))
                  (setq factorial (* factorial (+ i 1)))
                  (setq i (+ i 1)))))))))


;; Fix: Erf関数の精度が原因で正規分布X
;; expotential -> ziggurat
;; Randn -> ziggurat
;; gamma -> ziggurat
;; beta -> beta-bc + beta-cc
;; chisquare -> ziggurat
;; normal -> mgl-mat? numcl?
(eval-when (:compile-toplevel :load-toplevel :execute)
  (macrolet ((define-ziggurat-sampler (name pdf cdf ipd)
	       `(defun ,name (dtype &key (table-size *table-size*))
		  (let ((sampler (ziggurat-make-table dtype)))
		    (multiple-value-bind (rx ry ir) (funcall sampler ,pdf ,cdf ,ipd :table-size table-size)
		      (let ((generator (make-ziggurat-sampler dtype)))
			
			#'(lambda ()
			    (funcall generator rx ry ir table-size))))))))

    ;; FixMe
    (define-ziggurat-sampler make-normal-sampler
	#'(lambda (x) (exp (- (/ (* x x) 2)))) ;; PDF, -1/2*x^2/sqrt(2pi)
;;      '(lambda (x) (* (sqrt (* 2 pi))
;;		       0.5
;;		       (1+ (erf (/ x (sqrt 2))))))
      #'(lambda (x) (/ (1+ (erf (/ x (sqrt 2)))) 2))
      #'(lambda (y) (sqrt (* -2.0 (log y))))) ;; y = (0, 1) y -> +0, F -> 0 y -> -0, F -> ∞

    
    (define-ziggurat-sampler make-randn-sampler ;; PDF, -1/2*x^2/sqrt(2pi)
	#'(lambda (x) (/ (exp (- (/ (* x x) 2))) (sqrt (* 2 pi))))
      #'(lambda (x) (/ (1+ (erf (/ x (sqrt 2)))) 2))
      #'(lambda (y) (sqrt (* -2.0 (log y))))) ;; y = (0, 1) y -> +0, F -> 0 y -> -0, F -> ∞
    (define-ziggurat-sampler make-expotential-sampler
	#'(lambda (x) (exp (- x)))
      #'(lambda (x) (- 1 (exp (- x))))
      #'(lambda (y) (- (log y))))

    ))

