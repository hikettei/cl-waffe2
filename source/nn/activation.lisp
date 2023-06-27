
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

(defun !softmax (x)
  "
## [function] !softmax
"
  (let* ((x1 (!sub x (!mean x  :axis 1 :keep-repeat t)))
	 (z  (!sum   (!exp x1) :axis 1 :keep-repeat t)))
    (!div (!exp x1) z)))
