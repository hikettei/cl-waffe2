(in-package :cl-user)

(defpackage :mnist-example
  (:use
   :cl
   :cl-waffe2
   :cl-waffe2/nn
   :cl-waffe2/distributions
   :cl-waffe2/base-impl
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes))

(in-package :mnist-example)

;; To ADD:
;; call->
;; sequence
;; deftrainer

;; TO ADD:
;; (call-> x layer1 layer2 layer3)
;; (x = (sequence (LinearLayer1 (LinearLayer2 ...

;; ugly...
;; call -> Node
(defmodel (MLP-Model (self)
	   :slots ((layer1 :initarg :layer1 :accessor mlp-layer1)
		   (layer2 :initarg :layer2 :accessor mlp-layer2)
		   (layer3 :initarg :layer3 :accessor mlp-layer3))
	   :initargs (:layer1 (LinearLayer 784 512)
		      :layer2 (LinearLayer 512 256)
		      :layer3 (LinearLayer 256 10))
	   :on-call-> ((self x)
		       (call (mlp-layer3 self)
			     (!tanh (call (mlp-layer2 self)
					  (!tanh (call (mlp-layer1 self) x))))))))


(let ((model (MLP-Model)))
  (with-build (fw bw v p) (!sum (call model (randn `(10 784))))
    (time (funcall fw))
    (time (funcall bw))
    (time (funcall fw))
    (time (funcall bw))))


