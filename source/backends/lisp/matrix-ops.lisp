
(in-package :cl-waffe2/backends.lisp)

;; ArgMax
(define-with-typevar (max-kernel u) (x out
				     offset size incx
				     o-index)
  (declare (optimize (speed 3))
           (type (simple-array u (*)) x)
	   (type (simple-array (unsigned-byte 32) (*)) out)
	   (type fixnum offset size incx o-index))
  (let ((maxtmp)
	(index 0))
    (dotimes (i size)
      (let ((out (aref x (+ offset (the fixnum (* incx i))))))
	(if maxtmp
	    (if (> out maxtmp)
		(and (setq maxtmp out)
		     (setq index i)))
	    (and (setq maxtmp out) (setq index i)))))
    (setf (aref out o-index) (the fixnum index))
    nil))

;; ArgMin
(define-with-typevar (min-kernel u) (x out
				     offset size incx
				     o-index)
  (declare (optimize (speed 3))
           (type (simple-array u (*)) x)
	   (type (simple-array (unsigned-byte 32) (*)) out)
	   (type fixnum offset size incx o-index))
  (let ((mintmp)
	(index 0))
    (dotimes (i size)
      (let ((out (aref x (+ offset (the fixnum (* incx i))))))
	(if mintmp
	    (if (< out mintmp)
		(and (setq mintmp out)
		     (setq index i)))
	    (and (setq mintmp out) (setq index i)))))
    (setf (aref out o-index) (the fixnum index))
    nil))

(defun expand-argmax-form (x out &aux (index (gensym)))
  (let ((kernel (max-kernel (dtype x))))
    `(let ((,index 0))
       ,(call-with-view
	 #'(lambda (x-view o-view)
	     `(progn
		(funcall
		 ,kernel
		 (tensor-vec ,x) (tensor-vec ,out)
		 ,(offset-of x-view 0) ,(size-of x-view 0) ,(stride-of x-view 0)
		 ,index)
		(incf ,index ,(stride-of o-view 0))))
	 `(,x ,out)
	 :at-least-dim 1
	 :force-order t))))

(defun expand-argmin-form (x out &aux (index (gensym)))
  (let ((kernel (min-kernel (dtype x))))
    `(let ((,index 0))
       ,(call-with-view
	 #'(lambda (x-view o-view)
	     `(progn
		(funcall
		 ,kernel
		 (tensor-vec ,x) (tensor-vec ,out)
		 ,(offset-of x-view 0) ,(size-of x-view 0) ,(stride-of x-view 0)
		 ,index)
		(incf ,index ,(stride-of o-view 0))))
	 `(,x ,out)
	 :at-least-dim 1
	 :force-order t))))

(define-impl (ArgMax-Node :device LispTensor)
	     :forward ((self x out)
		       `(progn
			  ,(expand-argmax-form x out)
			  ,out)))

(define-impl (ArgMin-Node :device LispTensor)
	     :forward ((self x out)
		       `(progn
			  ,(expand-argmin-form x out)
			  ,out)))

;; MaxValue-Node
(define-with-typevar (max-value-kernel u)
    (x out
     offset size incx
     o-index)
  (declare (optimize (speed 3))
           (type (simple-array u (*)) x out)
	   (type fixnum offset size incx o-index))
  (let ((maxtmp))
    (dotimes (i size)
      (let ((out (aref x (+ offset (the fixnum (* incx i))))))
	(if maxtmp
	    (when (> out maxtmp)
	      (setq maxtmp out))
	    (setq maxtmp out))))
    (setf (aref out o-index) (the u maxtmp))
    nil))

(define-with-typevar (min-value-kernel u)
    (x out
     offset size incx
     o-index)
  (declare (optimize (speed 3))
           (type (simple-array u (*)) x out)
	   (type fixnum offset size incx o-index))
  (let ((mintmp))
    (dotimes (i size)
      (let ((out (aref x (+ offset (the fixnum (* incx i))))))
	(if mintmp
	    (when (< out mintmp)
	      (setq mintmp out))
	    (setq mintmp out))))
    (setf (aref out o-index) (the u mintmp))
    nil))

(defun expand-max-value-form (x out &aux (index (gensym)))
  (let ((kernel (max-value-kernel (dtype x))))
    `(let ((,index 0))
       ,(call-with-view
	 #'(lambda (x-view o-view)
	     `(progn
		(funcall
		 ,kernel
		 (tensor-vec ,x) (tensor-vec ,out)
		 ,(offset-of x-view 0) ,(size-of x-view 0) ,(stride-of x-view 0)
		 ,index)
		(incf ,index ,(stride-of o-view 0))))
	 `(,x ,out)
	 :at-least-dim 1
	 :force-order t))))


(defun expand-min-value-form (x out &aux (index (gensym)))
  (let ((kernel (min-value-kernel (dtype x))))
    `(let ((,index 0))
       ,(call-with-view
	 #'(lambda (x-view o-view)
	     `(progn
		(funcall
		 ,kernel
		 (tensor-vec ,x) (tensor-vec ,out)
		 ,(offset-of x-view 0) ,(size-of x-view 0) ,(stride-of x-view 0)
		 ,index)
		(incf ,index ,(stride-of o-view 0))))
	 `(,x ,out)
	 :at-least-dim 1
	 :force-order t))))

(define-impl (MaxValue-Node :device LispTensor)
	     :forward ((self x out)
		       `(progn
			  ,(expand-max-value-form x out)
			  ,out)))

(define-impl (MinValue-Node :device LispTensor)
	     :forward ((self x out)
		       `(progn
			  ,(expand-min-value-form x out)
			  ,out)))

