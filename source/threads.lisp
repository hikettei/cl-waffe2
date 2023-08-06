
(in-package :cl-user)

(defpackage :cl-waffe2/threads
  (:use :cl :lparallel)
  (:export
   #:*num-cores*
   #:*multithread-threshold*
   
   #:with-num-cores
   #:multithread-p

   #:maybe-with-lparallel

   #:maybe-pfuncall
   #:maybe-pdotimes))

(in-package :cl-waffe2/threads)

;; MultiThreading configs

(defparameter *num-cores* 1 "
## [parameter] *num-cores*

Indicates the number of cpu cores. Set 1 to disable all multithreading in cl-waffe2.")

(defparameter *multithread-threshold* 80000)
(declaim (type (unsigned-byte 64) *num-cores* *multithread-threshold*))

(defparameter *under-multi-thread* nil)

(defmacro with-num-cores ((num-core) &body body)
  "
## [macro] with-num-cores

```lisp
(with-num-cores (num-core) &body body)
```

Set *num-core*=num-core under the body execution.
"
  `(let ((*num-cores* ,num-core)
	 (*under-multi-thread* (or *under-multi-thread* (= ,num-core 1))))
     (maybe-with-lparallel ,@body)))

(defun multithread-p ()
  (not (<= (the fixnum *num-cores*) 1)))

(defmacro maybe-with-lparallel (&body body)
  `(let ((*kernel* (or *kernel* (if (multithread-p)
				    (make-kernel *num-cores*)
				    nil))))
     ,@body))

(defmacro maybe-pfuncall (function &rest args)
  `(if (multithread-p)
       (pfuncall ,function ,@args)
       (funcall ,function ,@args)))

;; kaettara test
;; optimize compared to pure dotimes.
;; gemm ... (with-num-cores (1) ...) is must.
;; gensym name conflicts?
(defmacro maybe-pdotimes ((var count) &body body &aux (thread-idx (gensym)))
  "
## [macro] maybe-pdotimes
"
  (let ((*under-multi-thread* t))
    (alexandria:with-gensyms (multi-thread-subject count-per-thread multi-thread-part from to)
      `(flet ((,multi-thread-subject (,from ,to)
		(declare (type (unsigned-byte 64) ,from ,to))
		(loop with *under-multi-thread* = t
		      for ,var fixnum upfrom ,from below ,to
		      do ,@body)))
	 (if (or *under-multi-thread*
		 (= *num-cores* 1)
		 (< (the fixnum ,count) *multithread-threshold*)
		 ;; [todo] benchmark
		 (< (the fixnum ,count) *num-cores*))
	     ;; If *num-cores* = 1 or count is enough small. ignore parallelize.
	     (,multi-thread-subject 0 ,count)
	     (maybe-with-lparallel
	       (let* ((,count-per-thread  (floor (/ (the fixnum ,count) *num-cores*)))
		      (,multi-thread-part (* *num-cores* ,count-per-thread)))
		 (declare (type (unsigned-byte 64) ,count-per-thread ,multi-thread-part))
		 (pdotimes (,thread-idx *num-cores*)
		   (locally (declare (type (unsigned-byte 64) ,thread-idx))
		     (,multi-thread-subject
		      (the fixnum (* ,thread-idx ,count-per-thread))
		      (the fixnum (* (1+ ,thread-idx) ,count-per-thread)))))
		 (,multi-thread-subject
		  ,multi-thread-part
		  ,count))))))))


