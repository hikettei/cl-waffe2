
(in-package :cl-waffe2/nn)

;; Softmax
;; ReLU
;; GeLU
;; Leakey-ReLU
;;

(defun !relu (x)
  "
## [function] !relu

..."

  (!mul x (A>scal x 0.0)))

(defun !gelu (x)
  "
## [function] !gelu

"
  
  )

;; todo (!matmul !t !t) test
;; Bug: (Proceed (!sum (Proceed (!Softmax x))))
(defun !softmax (x &key (avoid-overflow t))
  "
## [function] !softmax

Returns a tensor that normalizes the given `x` with computing `Softmax`

(TODO)
"

  (if avoid-overflow
      (let* ((x1 (!sub x (!mean x  :axis 1 :keepdims t)))
	     (z  (!sum   (!exp x1) :axis 1 :keepdims t)))
	(!div (!exp x1) z))
      (!div (!exp x) (!sum (!exp x) :axis 1 :keepdims t))))

