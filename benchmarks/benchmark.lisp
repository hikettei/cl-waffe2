
(in-package :cl-waffe2/benchmark)

(defmacro bench ((name
		  allocator
		  &key
		    (stream t)  ;; the place to print outputs
		    (n 100)     ;; try-n
		    (devices)   ;; using devices
		    (scales)    ;; scales
		    (backward)) ;; if t, include bw time
		 &body body)
  `(labels ((allocator (n)
	      (multiple-value-list ,allocator))
	    (building (,@(caar body))
	      (let ((out (progn ,@(cdar body))))
		(build out))))
     (with-memory-pool
       (format t "~%[INFO] Benchmarking ~a~%" ,name)
       (format
	,stream
	"~a"
	(with-output-to-string (out)
	  (format out "~%~%# ~a~%~%`backward=~a`" ,name ,backward)
	  (dolist (device ,devices)
	    (format out "~%~%## ~a scales=~a~%~%`" device ,scales)
	    (dolist (scale ,scales)
	      (format t "device=~a scale=~a~%" device scale)
	      #+sbcl(sb-ext:gc :full t)
	      (let ((*using-backend* `(,device))
		    (model (apply #'building (allocator scale))))
		;; Wall time
		(let ((t1 (get-internal-real-time)))
		  (dotimes (i ,n)
		    (forward model)
		    ,(if backward
			 `(backward model)))
		  (let* ((t2 (get-internal-real-time))
			 (mean-time (/ (float (/ (- t2 t1) internal-time-units-per-second)) ,n)))
		    (format out "~a, " mean-time)))))
	    (format out "`")))))))


(defun perform-math-bench (&key
			     (stream t)
			     (scales `(2 4 8 16 32 64 128 256 512 1024 2048))
			     (n 100)
			     (devices `(cl-waffe2/backends.lisp:LispTensor cl-waffe2/backends.jit.cpu:JITCPUTensor))
			     (backward nil))

  (macrolet ((bench-on (name function)
	     `(bench (,name (values (ax+b `(,n ,n) 0 1))
		      :stream stream
		      :n n
		      :devices devices
		      :backward backward
		      :scales scales)
		((x) (,function x)))))

    (bench-on "abs" !abs)
    
    (bench-on "sin" !sin)
    (bench-on "cos" !cos)
    (bench-on "tan" !tan)

    (bench-on "exp" !exp)
    (bench-on "log" !log)))


