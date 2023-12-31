
(in-package :cl-waffe2/vm.iterator)

;;
;; Iteration Space { X[a, b, c, d...] = op(Y[a, b, c, d, ...])}
;; a, b, c, d,に対して線形計画法で色々最適化 (Dependenciesを崩さない)
;;
;;

;;
;; Constraints
;; 1. Ranks are the same, upper/lower bounds are determined.
;; 2. Iterations must be increased by one.
;;

;; Polyhedral Compiler:
;; Iteration_Space
;;  i - A[i, j]
;;  |         X X
;;  |       X X X
;;  |     X X X X
;;  |   X X X X X
;;  | X X X X X X
;;  |------------- j
;;



