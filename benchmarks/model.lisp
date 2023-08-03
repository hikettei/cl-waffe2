
(in-package :cl-waffe2/benchmark)


(defun perform-model-bench-set (&key
				  (stream t)
				  (scales `(2 4 8 16 32 64 128 256 512 1024 2048))
				  (n 100)
				  (devices `(cl-waffe2/backends.lisp:LispTensor cl-waffe2/backends.jit.cpu:JITCPUTensor))
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
    
    (bench-on "with_no_grad softmax(x)" (with-no-grad (!softmax x)))))

(defun start-model-bench ()
  (with-open-file (stream "./benchmarks/results/model.md"
			  :direction :output
			  :if-exists :supersede
			  :if-does-not-exist :create)
    (perform-model-bench-set
       :stream stream
       :backward nil)))

