
(in-package :cl-waffe2/benchmark)


;; [FixME] In my poor CPU, macbook 2017 intel i5, can't receive the benefits of lparallel?
;; need more benchmarks...


;; lparallel benchmark
(defun perform-multi-threading-bench (&key
					(stream t)
					(scales `(1000 1500 2000 2500 3000))
					(n 100)
					(num-cores 1)
					(devices `(cl-waffe2/backends.cpu:CPUTensor))
					(backward nil))
  (macrolet ((bench-on (name &body body)
	       `(bench (,name (values (randn `(,n ,n)))
			:stream stream
			:n n
			:devices devices
			:more-devices (list 'cl-waffe2/backends.lisp:LispTensor)
			:backward backward
			:scales scales)
		  ((x) ,@body))))

    (cl-waffe2/vm.generic-tensor:reset-compiled-function-cache!)

    (cl-waffe2/threads:with-num-cores (num-cores)
      (bench-on (format nil "Sum(x) num-cores=~a" num-cores)
		(!sum x)))

    (cl-waffe2/threads:with-num-cores (num-cores)
      (bench-on (format nil "sin(x) num-cores=~a" num-cores)
		(!sin x)))))
		
(defun start-multi-threading-bench ()
  (with-open-file (stream "./benchmarks/results/lparallel.md"
			  :direction :output
			  :if-exists :supersede
			  :if-does-not-exist :create)
    (perform-multi-threading-bench
     :stream stream
     :num-cores 1
     :backward nil)
    (perform-multi-threading-bench
     :stream stream
     :num-cores 2
     :backward nil)
    (perform-multi-threading-bench
     :stream stream
     :num-cores 3
     :backward nil)
    (perform-multi-threading-bench
     :stream stream
     :num-cores 4
     :backward nil)))

;;(start-multi-threading-bench)
