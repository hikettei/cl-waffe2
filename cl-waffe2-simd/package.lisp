
(in-package :cl-user)

(defpackage :cl-waffe2-simd
  (:use :cl :cffi)
  (:export #:try-loading-simd-extension #:make-fname #:make-im2col #:make-col2im))

(in-package :cl-waffe2-simd)

