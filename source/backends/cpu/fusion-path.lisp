
(in-package :cl-waffe2/backends.cpu)

#|
(defpath (ReLU-CPU-Specialized-Fusion-No-Grad
	  (make-query 'Where-Operation-Node :device 'CPUTensor :dtype 'float)
	  (make-query 'MulNode              :device 'CPUTensor :dtype 'float))
	 :reject-p #'simd-extension-p
	 :replaced-with ((self where mul)

			 ))
|#

