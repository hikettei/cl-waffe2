
(in-package :cl-waffe2/benchmark)

(defmacro bench ((name
		  allocator
		  &key
		    (stream t)  ;; the place to print outputs
		    (n 100)     ;; try-n
		    (devices)   ;; using devices
		    (more-devices)
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
	      (let ((*using-backend* (append `(,device)
					     ,more-devices))
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

