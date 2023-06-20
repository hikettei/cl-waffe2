
(in-package :cl-waffe2/distributions)

;; uniform-random
;; fixnum-random
;; random with the range
;;


(defun sample-uniform-random (from to)
  "x = [from to)"
  (+ (random (abs (- from to))) from))

;; bernoulli
(defun sample-bernoulli (p)
  "p -> 1.0 (1-p) -> 0.0"
  (if (< (random 1.0) p)
      1
      0))
