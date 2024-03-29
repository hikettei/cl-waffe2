
(in-package :cl-waffe2/distributions)

;; = References ===============================================
;; https://andantesoft.hatenablog.com/entry/2023/04/30/183032
;; Marsaglia, G., & Tsang, W. W. (2000). The ziggurat method for generating random variables. Journal of statistical software.
;; https://marui.hatenablog.com/entry/2023/01/23/194507
;; ============================================================

(defparameter *table-size* 256)

;; FIXME: Pointer Coercion of double-float
;; Could be optimized much more, by using ash/logand
(define-with-typevar-dense (make-ziggurat-table u) (pdf ipd r v)
  (declare ;;(optimize (safety 0))
	   (type function pdf ipd)
	   (type double-float r v))
  (let ((r (coerce r (quote u))) ;; x0
	(v (coerce v (quote u))) ;; area0
	(rx (make-array *table-size* :element-type  (quote u)))
	(ry (make-array *table-size* :element-type  (quote u)))
	(irr (make-array *table-size* :element-type (quote u)))
	(zero (coerce 0 (quote u))))
    (declare (type u r v)
	     (type fixnum *table-size*))
    
    (setf (aref rx 0)
	  (/ v (the u (funcall pdf r))))

    (setf (aref rx 1) r)

    (loop for i fixnum upfrom 2 below *table-size*
	  do (setf (aref rx i)
		   (the u
			(funcall ipd (+ (/ v (aref rx (1- i)))
					(funcall pdf (aref rx (1- i)))))))
	  if (< (aref rx i) zero)
	    do (error "xi < 0, parameters may be wrong"))

    (loop for i fixnum upfrom 0 below (1- *table-size*)
	  if (= (1+ i) *table-size*)
	    do (setf (aref ry i) (the u (funcall pdf zero)))
	  else
	    do (setf (aref ry i) (the u (funcall pdf (aref rx (1+ i))))))
    
    (loop for i fixnum upfrom 0 below (1- *table-size*)
	  do (setf (aref irr i) (/ (aref rx (1+ i))
				   (aref rx i))))
    (values rx ry irr)))

(define-with-typevar-dense (make-gaussian-generator u) (rx ry ir table-size)
  (declare (optimize (speed 3) (safety 0))
   (type (simple-array u (*)) rx ry ir)
   (type fixnum table-size))
  (let ((one (coerce 0.9999999 (quote u)))
	(two (coerce 2 (quote u))))
    (loop while t
	  ;; Ugly: using (random) for several times.
	  ;; Proceed with double-first at first
	  do (let* ((u1f (- (random two) one)) ;; [-1, 1)
		    (i   (random table-size))) ;; ugly part
	       (declare (type u u1f)
			(type fixnum i))
	       (when (< (abs u1f)
			(aref ir i))
		 (return (* u1f (aref rx i))))

	       (if (= i 0)
		   (let ((s  two)
			 (t1 one)
			 (x0 (aref rx 1)))
		     ;; Iter Until out of range
		     (loop while (> (* s s) (* t1 t1))
			   do (and
			       (setq s  (- (/ (log (- 1 (random one))) x0)))
			       (setq t1 (/ (log (- 1 (random one))) x0)))))
		   (let ((x (* u1f (aref rx i)))
			 (y (- (random two) one)))
		     (declare (type u x y))
		     (if (<=
			  (+ (aref ry (1- i))
			     (* y
				(-
				 (aref ry i)
				 (aref ry (1- i)))))
			  (exp (* -0.5 x x)))
			 (return x))))))))

(define-with-typevar-dense (make-exponential-generator u) (rx ry ir table-size)
  (declare (optimize (speed 3) (safety 0))
   (type (simple-array u (*)) rx ry ir)
   (type fixnum table-size))
  (let ((one  (coerce 0.9999999 (quote u)))
	(two  (coerce 2 (quote u)))
	(tail (coerce 0 (quote u))))
    (declare (type u one two tail))
    (loop while t
	  ;; Proceed with double-first at first
	  do (let* ((u1f (random one)) ;; [0, 1)
		    (i   (random table-size))) ;; ugly
	       (declare (type u u1f)
			(type fixnum i))
	       (when (< (abs u1f)
			(aref ir i))
		 (return (+ tail (* u1f (aref rx i)))))

	       (if (= i 0)
		   (let ((x0 (aref rx 1)))
		     (incf tail x0))
		   (let ((x (* u1f (aref rx i)))
			 (y (- (random two) one)))
		     (declare (type u x y))
		     (if (<=
			  (+ (aref ry (1- i))
			     (* y
				(-
				 (aref ry i)
				 (aref ry (1- i)))))
			  (exp (- x)))
			 (return (+ tail x)))))))))

;; =========================================================================
;; First-time-call  -> there's a little overhead because creates lut
;; Second-time-call -> No overhead
;; To reduce memory-space, the LUT won't be created until the distribution at specified dtype IS BEING SAMPLED.
;; =========================================================================
(eval-when (:compile-toplevel :load-toplevel :execute)
  (macrolet ((define-ziggurat-sampler (getter-name name pdf ipd r v sampler1)
	       (let ((params (loop for dtype in `(:float :double)
				   collect (symb '* getter-name (intern (symbol-name dtype)) '*))))
		 `(progn
		    (defun ,name (dtype)
		      (let ((sampler (make-ziggurat-table dtype)))
			(multiple-value-bind (rx ry ir)
			    (funcall sampler ,pdf ,ipd ,r ,v)
			  (let ((generator (,sampler1 dtype)))
			    #'(lambda ()
				(funcall generator rx ry ir *table-size*))))))

		    ,@(loop for p in params
			    collect `(defparameter ,p nil))
		    
		    (defun ,getter-name (dtype)
		      (case dtype
			,@(loop for pname in params
				for dtype in `(:float :double)
				collect `(,dtype
					  (if (null ,pname)
					      (setf ,pname (,name ,dtype))
					      ,pname)))))))))

    (define-ziggurat-sampler get-randn-sampler
      make-randn-sampler ;; PDF, -1/2*x^2/sqrt(2pi)
      #'(lambda (x) (exp (- (/ (* x x) 2.0))))
      #'(lambda (y) (sqrt (* -2.0 (log y))))
      3.6541528853613281d0
      0.0049286732339721695d0
      make-gaussian-generator)
    
    (define-ziggurat-sampler get-exponential-sampler
      make-exponential-sampler
      #'(lambda (x) (exp (- x)))
      #'(lambda (y) (- (log y)))
      7.6971174701310288d0
      0.0039496598225815527d0
      make-exponential-generator)))


