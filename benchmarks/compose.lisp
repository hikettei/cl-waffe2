
(in-package :cl-waffe2/benchmark)

(defun perform-composed-bench-set (&key
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

    (bench-on "sin(sin(sin x))"
	      (!sin (!sin (!sin x))))

    (bench-on "ReLU(x)"	   (!relu x))
    (bench-on "sigmoid(x)" (!sigmoid x))
    (bench-on "softmax(x)" (!softmax x))))

(defun start-composed-bench ()
  (with-open-file (stream "./benchmarks/results/composed_ops.md"
			  :direction :output
			  :if-exists :supersede
			  :if-does-not-exist :create)
    (perform-composed-bench-set
     :stream stream
     :backward nil)
    (perform-composed-bench-set
     :stream stream
     :backward t)))


;; 追加すること
;;  LinearLayer
;;  sb-profileをするためのベンチマーク
;;  JITCompilerのベンチマーク (Softmaxは0.1s以内にコンパイルしたい！MLPも)
;;

