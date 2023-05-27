
(in-package :cl-waffe2/distributions)


;; Ref: https://dl.acm.org/doi/pdf/10.1145/359460.359482


(define-with-typevar-dense (beta-bb u) (a0 a b)
  (declare (optimize (speed 3) (safety 0))
	   (type u a0)
	   (type (u 0e0) a b))

  (unless (>= (min a b) 1.0)
    (error "cl-waffe:!beta failed because of (min a b) > 1."))

  (let* ((alpha (+ a b))
  	 (beta  (sqrt (the (u 0e0)
			   (/ (- alpha 2.0)
			      (- (* 2.0 a b) alpha)))))
	 (gamma (+ a (/ beta)))
	 (r0 0.0)
	 (w0 0.0)
	 (t0 0.0))
    (labels ((next (&aux
		      (u1 (random 1.0))
		      (u2 (random 1.0))
		      (v (* beta (- (log u1) (log (+ 1.0 (- u1)))))))
	       (declare (type u u1 u2 v))
	       
	       (setq w0 (* a (exp v)))
	       (setq r0 (- (* gamma v) 1.3862944))
	       
	       (let* ((z (* u1 u1 u2))
		      (s (+ a r0 (- w0))))
		 (declare (type u z s))
		 
		 (if (>= (+ s 2.609438) (* 5 z))
		     nil
		     (progn
		       (setq t0 (log z))
		       (if (>= s t0)
			   nil
			   t))))))

      (loop while (and
		   (next)
		   (< (+ r0
			 (* alpha (- (log alpha) (log (+ b w0)))))
		      t0)))

      (if (= a a0)
	  (/ w0 (+ b w0))
	  (/ b (+ b w0))))))


;; FIXME: Unstable
(define-with-typevar-dense (beta-bc utype) (a0 a b)
  (declare (optimize (speed 3) (safety 0))
	   (type utype a0)
	   (type (utype 0e0) a b))

  (unless (<= (min a b) 1.0)
    (error "cl-waffe:!beta failed because of (min a b) <= 1."))

  (let* ((alpha (+ a b))
  	 (beta  (/ b))
	 (gamma (+ 1 a (- b)))
	 (k1 (* gamma (+ 0.0138889 (* 0.0416667 b)) (/ (+ (* a beta) -0.777778))))
	 (k2 (+ 0.25 (* b (+ 0.5 (/ 0.258 gamma)))))
	 (z  0.0)
	 (y  0.0)
	 (v 0.0)
	 (w 0.0)
	 (f t)
	 (u1 0.0)
	 (u2 0.0)
	 (lp t))
    (declare (type utype alpha beta gamma k1 k2 z y w v u1 u2)
	     (type boolean lp f))
    
    (labels ((next ()
	       (setq lp t)
	       (setq u1 (random 1.0))
	       (setq u2 (random 1.0))
	       (if (>= u1 0.5)
		   (progn
		     (setq z (* u1 u1 u2))
		     (if (<= z 0.25)
			 (progn
			   (setq v (* beta
				      (the utype
					   (- (log u1) (log (1+ (- u1)))))))
			   (setq w (* a (exp v)))
			   nil)
			 (if (>= z k2)
			     (progn
			       (setq lp nil)
			       t)
			     t)))
		   (progn
		     (setq y (* u1 u2))
		     (setq z (* u1 y))
		     (if (>= (+ (* 0.25 u2) z (- y)) k1)
			 (progn
			   (setq lp nil)
			   t)
			 t)))))

      (loop while (and f (next))
	    do (when lp
		 (setq v (* beta
			    (the utype
				 (- (log u1) (log (1+ (- u1)))))))
		 (setq w (* a (exp v)))
		 (if (>= (- (* alpha
			       (+ v
				  (log alpha)
				  (- (log (1+ (+ b w))))))
			    1.3862944)
			 (log z))
		     (setq f nil))))

      (if (= a a0)
	  (/ w (+ b w))
	  (/ b (+ b w))))))


;; chisquare
;; gamma
;; etc...

