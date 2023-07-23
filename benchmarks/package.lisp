
(in-package :cl-user)

(ql:quickload :clgplot)
(ql:quickload :cl-waffe)

(load "./cl-waffe2.asd")
(ql:quickload :cl-waffe2)

(defpackage :cl-waffe2/benchmark
  (:use :cl
	:clgplot
	:cl-waffe2
	:cl-waffe2/vm.generic-tensor
        :cl-waffe2/vm.nodes
        :cl-waffe2/base-impl
   :cl-waffe2/distributions))

(in-package :cl-waffe2/benchmark)

(defun cl-waffe2-bench (n &key (times 1000))
  (with-no-grad
    (let ((model (build (!matmul (randn `(,n ,n))
				 (randn `(,n ,n)))
			:compile-mode :fastest)))
      (declare (type Compiled-Composite model))
      (let ((t1 (get-internal-real-time)))
	(dotimes (i times)
          (forward model))
	(let ((t2 (get-internal-real-time)))
	  (float (/ (- t2 t1) internal-time-units-per-second)))))))

(defun cl-waffe-bench (n &key (times 1000))
  (cl-waffe:with-no-grad
    (let ((x (cl-waffe:!randn `(,n ,n)))
	  (y (cl-waffe:!randn `(,n ,n))))
      (let ((t1 (get-internal-real-time)))
	(dotimes (i times)
	  (cl-waffe:!matmul x y))
	(let ((t2 (get-internal-real-time)))
	  (float (/ (- t2 t1) internal-time-units-per-second)))))))

(defun cl-waffe-bench-in-place (n &key (times 1000))
  (cl-waffe:with-no-grad
    (let ((x (cl-waffe:data (cl-waffe:!randn `(,n ,n))))
	  (y (cl-waffe:data (cl-waffe:!randn `(,n ,n))))
	  (out (cl-waffe:data (cl-waffe:!zeros `(,n ,n)))))
      (let ((t1 (get-internal-real-time)))
	(dotimes (i times)
	  (mgl-mat:gemm! 1.0 x y 0.0 out))
	(let ((t2 (get-internal-real-time)))
	  (float (/ (- t2 t1) internal-time-units-per-second)))))))

(defun measure-bench (function from to by times)
  (values
   (loop for N fixnum upfrom from to to by by
	 collect N)
   (loop for N fixnum upfrom from to to by by
	 do (format t "N = ~a ...~%" N)
	 collect (funcall function N :times times))))

(defun bench (functions
	      titles
	      &key
		(from 1)
		(to   300)
		(by   10)
		(times))
  (let ((x-seqs)
	(y-seqs)
	(titles-use))

    (mapc
     #'(lambda (title f)
	 (format t "[INFO] Benchmarking on ~a. from=~a to=~a by=~a~%"
		 f from to by)
	 ;;(sb-ext:gc :full t)
	 (multiple-value-bind (x y) (measure-bench f from to by times)
	   (push x x-seqs)
	   (push y y-seqs)
	   (push title titles-use)))
     titles functions)

    (plots (reverse y-seqs)
	   :x-seqs (reverse x-seqs)
	   :title-list (reverse titles-use)
	   :x-label "time (s)"
	   :y-label "Matrix Scale (N x N)"
	   :x-logscale t
	   :y-logscale t
	   :output "./assets/mm_bench_1000.png")

    (plots (reverse y-seqs)
	   :x-seqs (reverse x-seqs)
	   :title-list (reverse titles-use)
	   :x-label "time (s)"
	   :y-label "Matrix Scale (N x N)"
	   :output "./assets/mm_bench_1000_normal.png")))

(bench
 (list #'cl-waffe2-bench #'cl-waffe-bench #'cl-waffe-bench-in-place)
 (list "cl-waffe2" "cl-waffe" "cl-waffe (In-place)")
 :from 1
 :to 500
 :by 10
 :times 1000)

;; cl-waffe
;; cl-waffe (in-place)
;; cl-waffe2

