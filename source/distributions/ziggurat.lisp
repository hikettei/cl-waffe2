
(in-package :cl-waffe2/distributions)

;; = References ===============================================
;; https://andantesoft.hatenablog.com/entry/2023/04/30/183032
;; Marsaglia, G., & Tsang, W. W. (2000). The ziggurat method for generating random variables. Journal of statistical software.
;; https://marui.hatenablog.com/entry/2023/01/23/194507
;; ============================================================

;; TODO: Optimize
(defparameter *table-size* 256)
;; Generates a table.
;; TODD: Implement Modified Ziggurat

(defun ~= (x y)
  (declare (type number x y))
  (< (- x y) (typecase x
	       (single-float 0.0001)
	       (double-float 0.000001)
	       (T 0.001))))

;; No Need To Optimize
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
	       (+ (the u (* x (the u (funcall pdf x))))
		  (the u (funcall cdf +inf))
		  (- (the u (funcall cdf x)))))
	     (error-of (x)
	       (declare (type u x))
	       (let ((area (area-of x))
		     (xi x))
		 ;; Memo: pdfの値が(0.0, 1.0)にならない・・・
		 (loop for i fixnum upfrom 1 below table-size
		       do (progn
			    (setq xi (the u (funcall ipd (+ (/ area xi) (funcall pdf xi)))))
			    ;; Assertion, pdf >= 0
			    (when (< xi zero)
			      (return-from error-of -inf))))
		 ;; Check IsNAN? this is processing-system-dependant...
		 xi)))

      ;; Compute x0
      (let ((left  (coerce 2   (quote u)))
	    (right (coerce 10  (quote u)))
	    (mid   (coerce 6   (quote u)))
	    (half  (coerce 0.5 (quote u)))) ;; mid = 1/2(left+right)
	(declare (type u left right mid half))
	(loop named btree
	      while t
	      do (progn
		   (setq mid (* half (+ left right)))
		   (cond
		     ((and (= left mid)
			   (< mid right))
		      (setq mid (1+ left)))
		     ((and (= right mid)
			   (< left mid))
		      (setq mid (1- right)))
		     ((and (~= left mid)
			   (~= right mid))
		      (return-from btree)))
		   (let ((left-error  (error-of left))
			 (mid-error   (error-of mid))
			 (right-error (error-of right)))
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

      ;; (print x0)
      ;; (print x0-area)
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

;; Reference: https://marui.hatenablog.com/entry/2023/01/23/194507
(defun erfccheb (z)
  "Approximation of erfc function using Chebyshev method with precalculated coefficients."
  (assert (> z 0) nil "erfccheb requires nonnegative argument")
  (let* ((cof #(-1.3026537197817094d0 6.4196979235649026d-1 1.9476473204185836d-2 -9.561514786808631d-3 -9.46595344482036d-4 3.66839497852761d-4 4.2523324806907d-5 -2.0278578112534d-5 -1.624290004647d-6 1.303655835580d-6 1.5626441722d-8 -8.5238095915d-8 6.529054439d-9 5.059343495d-9 -9.91364156d-10 -2.27365122d-10 9.6467911d-11 2.394038d-12 -6.886027d-12 8.94487d-13 3.13092d-13 -1.12708d-13 3.81d-16 7.106d-15 -1.523d-15 -9.4d-17 1.21d-16 -2.8d-17))
         (d 0)
         (dd 0)
         (tt (/ 2 (+ 2 z)))
         (ty (- (* 4 tt) 2))
	 (tmp))
    (loop for j from (1- (length cof)) downto 1
          do (setq tmp d)
             (setq d (+ (* ty d) (* -1 dd) (aref cof j)))
             (setq dd tmp))
    (* tt (exp (+ (* -1 z z) (* 0.5 (+ (aref cof 0) (* ty d))) (- dd))))))

(defun erf (x &aux (inf 6.5d0)) ;; x -> 6.5d0, erf(x) -> 1.0d0
  "Error function."
  (if (or (> x inf) (< x (- inf)) (= x 0)) ;; If x is enough large, treated as +inf
      (signum x)
      (if (> x 0)
	  (- 1 (erfccheb x))
	  (- (erfccheb x) 1))))

;; erfでもいい気がする
(defun erf1 (x &aux (inf 10000.0)) ;; x -> 6.5d0, erf(x) -> 1.0d0
  "Error function."
  (if (or (> x inf) (< x (- inf)) (= x 0)) ;; If x is enough large, treated as +inf
      (signum x)
      (let ((erf (/ 2 (sqrt pi)))
	    (sum 0)
	    (fct 1))
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
		   
      

;; Fix: Erf関数の精度が原因で正規分布X
(eval-when (:compile-toplevel :load-toplevel :execute)
  (macrolet ((define-ziggurat-sampler (name pdf cdf ipd)
	       `(defun ,name (dtype &key (table-size *table-size*))
		  (let ((sampler (ziggurat-make-table dtype)))
		    (multiple-value-bind (rx ry ir) (funcall sampler ,pdf ,cdf ,ipd :table-size table-size)
		      (let ((generator (make-ziggurat-sampler dtype)))
			
			#'(lambda ()
			    (funcall generator rx ry ir table-size))))))))

    (define-ziggurat-sampler make-normal-sampler
	#'(lambda (x) (exp (* -1.0 x x 0.5))) ;; PDF, -1/2*x^2/sqrt(2pi)
      #'(lambda (x) (* (sqrt (* 2 pi))
		       0.5
		       (1+ (erf (/ x (sqrt 2))))))
      #'(lambda (y) (sqrt (* -2.0 (log y))))) ;; y = (0, 1) y -> +0, F -> 0 y -> -0, F -> ∞
    (define-ziggurat-sampler make-expotential-sampler
	#'(lambda (x) (exp (- x)))
      #'(lambda (x) (- 1 (exp (- x))))
      #'(lambda (y) (- (log y))))

    ))

