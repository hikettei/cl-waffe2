
(in-package :cl-waffe2/backends.lisp)

;; LazyXXXNodes are also dispatched by what functions are compiled
(defun lazy-dispatcher (self x y)
  (declare (ignore x y))
  `(,(forward-of self) ,(backward-of self) ,(reduced-of self)))

;; AbstractElementWise
(define-impl (Lazy-Function-Node :device LispTensor :cache-id #'lazy-dispatcher)
	     :forward ((self x out)
		       (lazy-call-form
			(list x out)
			(list
			 (make-lli
			  (forward-of self)
			  out
			  (list x)
			  :apply nil))
			out)))

(define-impl (Lazy-Index-Components-Node :device LispTensor :cache-id #'lazy-dispatcher)
	     :forward ((self x out)
		       (lazy-call-form
			(list x out)
			(list
			 (make-lli
			  (forward-of self)
			  out
			  (list x)
			  :apply nil))
			out
			:index-components t)))

(define-impl (Lazy-Reduce-Node :device LispTensor :cache-id #'lazy-dispatcher)
	     :forward ((self reduced x)
		       (lazy-call-form
			(list reduced x)
			(list
			 (make-lli
			  (forward-of self)
			  reduced
			  (list x)
			  :apply T
			  :reduced-to (reduced-of self)))
			reduced)))

