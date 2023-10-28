
(in-package :cl-user)

(defpackage :cl-waffe2/base-impl.test
  (:use :cl
        :fiveam
        :cl-waffe2
	:cl-waffe2/base-impl
        :cl-waffe2/vm.generic-tensor
	:cl-waffe2/distributions
        :cl-waffe2/vm.nodes))

(in-package :cl-waffe2/base-impl.test)

;; Add:
;; Testing Framework of all defnode
;; Proceed/Reshape/Squeeze etc...
;;
;;

(def-suite :base-impl-test)

(defparameter *dense-types*  `(:float :double))
(defparameter *sparse-types* `(:int8 :int16 :int32))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun symb (&rest inputs)
    (intern (with-output-to-string (out) (dolist (sym inputs) (princ sym out))))))

(defmacro define-tester (name op-type &body body)
  (declare (type (member :dense :sparse :all) op-type)
	   #+sbcl(sb-ext:muffle-conditions cl:style-warning))
  
  `(progn;;eval-when (:compile-toplevel :load-toplevel :execute)
     (export ',name)
     ;; You can use this macro for testing other backends, other dtypes.
     
     (defmacro ,name (&rest backend)
       #+sbcl(declare (sb-ext:muffle-conditions cl:style-warning))
       `(let ((*using-backend* '`(,,@backend)))
	  ,@(map 'list #'(lambda (dtype)
			   `(test ,(symb ', name '- dtype '- (car backend))
			      (is (with-dtype ,dtype
				    (progn;with-memory-pool
				      (let ((result (progn ,,@body)))
					(if (eql result t)
					    t
					    (if (eql result :backward)
						(error "the result of backward is invalid")
						(error "the result of forward is invalid")))))))))
		 ',(case op-type
		     (:sparse *sparse-types*)
		     (:dense  *dense-types*)
		     (:all `(,@*sparse-types* ,@*dense-types*))))))))

(defun M= (tensor1 tensor2)
  (every #'= (tensor-vec tensor1) (tensor-vec tensor2)))


(defun ~= (x y)
  (< (- x y) 0.00001))

