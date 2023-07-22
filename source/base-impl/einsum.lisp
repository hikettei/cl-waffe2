
(in-package :cl-waffe2/base-impl)

;; This file provides an superficial APIs of einsum:

;;
;; (defmacro %einsum %implict-einsum)
;;

;; 行列のShapeはわかってるか一部分だけわかってる (10 batch-size) みたいな

(defmacro %einsum (&body einsum-subscripts)
  "
## [macro] %einsum
"
  (multiple-value-bind (names-from names-to subs-from subs-to let-bindings) (parse-einsum-syntax `,einsum-subscripts)
    (let ((einsum (compile-einsum names-from names-to subs-from subs-to let-bindings)))
      `(let* (,@let-bindings)
	 ;; Place body
	 ))))

(defmacro %implict-einsum (&body einsum-subscripts)
  "
## [macro] %implict-einsum

"
  `(%einsum ,@einsum-subscripts -> [~]))

