
(in-package :cl-user)

(defpackage :cl-waffe2/nn
  (:nicknames #:wf/nn)
  (:use :cl :cl-waffe2 :cl-waffe2/distributions :cl-waffe2/base-impl :cl-waffe2/vm.nodes :cl-waffe2/vm.generic-tensor :cl-waffe2/threads)

  ;; LinearLayer
  (:export
   #:weight-of
   #:bias-of
   #:alpha-of
   #:beta-of)
  
  (:export
   #:LinearLayer
   #:Linear-weight
   #:Linear-bias)

  (:export
   #:BatchNorm2D
   #:LayerNorm2D)

  (:export
   #:Conv2D
   #:apply-conv2d
   #:!im2col
   #:unfold

   #:MaxPool2D
   #:AvgPool2D)

  ;; Criterions
  (:export
   #:L1Norm
   #:MSE
   #:cross-entropy-loss
   #:softmax-cross-entropy
   )

  ;; Non Linear Functions
  (:export
   #:!relu
   #:!leakey-relu
   #:!sigmoid
   #:!gelu
   #:!elu
   #:!softmax))

(in-package :cl-waffe2/nn)

