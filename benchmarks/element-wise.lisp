
(in-package :cl-waffe2/benchmark)


;; measure performance on element-wise function

(defun perform-math-bench-set (&key
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
    (bench-on "loge" !logE)))


(defun start-math-bench ()
  (with-open-file (stream "./benchmarks/results/math.md"
			  :direction :output
			  :if-exists :supersede
			  :if-does-not-exist :create)
    (perform-math-bench-set
     :stream stream
     :backward nil)
    (perform-math-bench-set
     :stream stream
     :backward t)))

;;(start-math-bench)
