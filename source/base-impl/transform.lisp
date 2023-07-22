
(in-package :cl-waffe2/base-impl)

;; Add a macro like:
;; (transform A[i j] -> A[i ~ j])
;;
;; Add a reader macro: #T

(defmacro %transform (&body subscripts)
  "
## [macro] %transform

%transform is a general-purpose UI to use `permute` and `broadcast`.

"
  (multiple-value-bind (names-from names-to subs-from subs-to let-bindings) (parse-einsum-syntax `,subscripts)

    ))

;;
;; (!add #T(a[i j] -> a[~ i j]) b)
;; (!add #T
;;       
;;
;;

;;(defun batched-matmul (A B)
;;  (!matmul (%transform A[i j] -> A[~ i j]) B))

;;(defun batched-matmul (A B)
;;  (!matmul #T(A[i j] -> A[~ i j]) B))
;;
;;#T(A[i j] -> A[!3 !100])
;; !100 -> broadcast
;; 0..10 -> Slice
;; 3 2 1 -> Index
;;
;; #T(A[~ i j] -> A[i j]) Unbroadcast/Setting broadcast again
;; #T(A[~ i j] -> A[i ~ j])
;; Inputに~があったらBroadcastableではないのに注意
;; (!add #T(A[~ i j] -> A[~ i i]) 0.0) ;; -> i vec is returned.
;;
;; 1 1 1               +--                 +
;; 1 1 1 -> Cutting -> -+- -> Resulting -> +
;; 1 1 1               --+                 +
;;
;; ↑だめ
;; einsumは無かったことにします。。。
;; -> 0 1 1, 1 0 1, 1 1 0... 
;; とかできたらどうだろう View == loop for
;;
;;

