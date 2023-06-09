
(in-package :cl-waffe2/backends.cpu)

;; ===============================================================
;; Level1
;; ===============================================================

(define-blas-function (axpy)
    (:void ((n :int) (alpha :float) (x :mat :input) (incx :int)
            (y :mat :io) (incy :int))))

(define-blas-function (copy)
    (:void ((n :int) (x :mat :input) (incx :int) (y :mat :output) (incy :int))))


;; ===============================================================
;; Level2
;; ===============================================================

;; sbmv
;; max
;; min

;; ===============================================================
;; Level3
;; ===============================================================


(define-blas-function (gemm)
    (:void ((transpose-a :char) (transpose-b :char) (n :int) (m :int) (k :int)
            (alpha :float) (a :mat :input) (lda :int)
            (b :mat :input) (ldb :int)
            (beta :float) (c :mat :io) (ldc :int))))

