
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

(deftype index () `(unsigned-byte 64))

(defvar *cl-waffe2-kernel* nil)
(defparameter *num-cores* 1 "
## [parameter] *num-cores*

Indicates the number of cpu cores. Set 1 to disable all multithreading in cl-waffe2.")

(defparameter *multithread-threshold* 80000)
(declaim (type index *num-cores* *multithread-threshold*))

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
  `(let* ((*cl-waffe2-kernel* (or *cl-waffe2-kernel* (if (multithread-p)
							 (make-kernel *num-cores*)
							 nil)))
	  (*kernel* *cl-waffe2-kernel*))
     ,@body))

(defmacro maybe-pfuncall (function &rest args)
  `(if (multithread-p)
       (pfuncall ,function ,@args)
       (funcall ,function ,@args)))

;; kaettara test
;; optimize compared to pure dotimes.
;; gemm ... (with-num-cores (1) ...) is must.

(defmacro maybe-pdotimes ((var count) &body body &aux (thread-idx (gensym)))
  "
## [macro] maybe-pdotimes
"
  (let ((*under-multi-thread* t))
    (alexandria:with-gensyms (multi-thread-subject count-per-thread multi-thread-part from to)
      `(flet ((,multi-thread-subject (,from ,to)
		(declare (type index ,from ,to))
		(loop with *under-multi-thread* = t
		      for ,var of-type index upfrom ,from below ,to
		      do ,@body)))
	 (if (or *under-multi-thread*
		 (= (the fixnum *num-cores*) 1)
		 (< (the index ,count) *multithread-threshold*)
		 ;; [todo] benchmark
		 (< (the index ,count) *num-cores*))
	     ;; If *num-cores* = 1 or count is enough small. ignore parallelize.
	     (,multi-thread-subject 0 (the index ,count))
	     (maybe-with-lparallel
	       (let* ((,count-per-thread  (floor (the fixnum ,count) (the index *num-cores*)))
		      (,multi-thread-part (* (the index *num-cores*) ,count-per-thread)))
		 (declare (type index ,count-per-thread ,multi-thread-part))
		 (pdotimes (,thread-idx *num-cores*)
		   (locally (declare (type index ,thread-idx))
		     (,multi-thread-subject
		      (the index (* ,thread-idx ,count-per-thread))
		      (the index (* (1+ ,thread-idx) ,count-per-thread)))))
		 (,multi-thread-subject
		  ,multi-thread-part
		  (the index ,count)))))))))


