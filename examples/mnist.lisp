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

;; one more ogly part: defmodel

(defmodel (MLP-Model (self)
	   :slots ((layer1 :initarg :layer1 :accessor mlp-layer1)
		   (layer2 :initarg :layer2 :accessor mlp-layer2)
		   (layer3 :initarg :layer3 :accessor mlp-layer3))
	   :initargs (:layer1 (LinearLayer 784 512)
		      :layer2 (LinearLayer 512 256)
		      :layer3 (LinearLayer 256 10))
	   :on-call-> ((self x)
		       (call-> x
			       (slot-value self 'layer1)
			       (asnode #'!tanh)
			       (slot-value self 'layer2)
			       (asnode #'!tanh)
			       (slot-value self 'layer3)
			       (asnode #'!softmax)))))

(defsequence MLP-Sequence (in-features hidden-dim out-features
			   &key (activation #'!tanh))
	     "3 Layers MLP"
	     (LinearLayer in-features hidden-dim)
	     (asnode activation)
	     (LinearLayer hidden-dim hidden-dim)
	     (asnode activation)
	     (LinearLayer hidden-dim out-features)
	     (asnode #'!softmax))

(defun train ()
  (let ((model (MLP-Sequence 784 256 10)))
    (with-build (forward backward vars params)
		(!sum (call model (make-input `(batch-size 784) :X)))
      (embody-input vars :X (randn `(10 784)))

      (time (print (funcall forward)))
      (time (funcall backward))
      
      )))
