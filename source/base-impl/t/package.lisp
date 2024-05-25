
(in-package :cl-user)

(defpackage :cl-waffe2/base-impl.test
  (:use :cl
	:rove
        :cl-waffe2
        :cl-waffe2/backends.lisp
        :cl-waffe2/backends.cpu
	:cl-waffe2/base-impl
        :cl-waffe2/vm.generic-tensor
	:cl-waffe2/distributions
        :cl-waffe2/vm.nodes))

(in-package :cl-waffe2/base-impl.test)

(defparameter *dense-types*  `(:float :double))
(defparameter *sparse-types* `(:int8 :int16 :int32))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun symb (&rest inputs)
    (intern (with-output-to-string (out) (dolist (sym inputs) (princ sym out))))))

(defmacro define-tester (name op-type &body body)
  (declare (type (member :dense :sparse :all) op-type)
	   #+sbcl(sb-ext:muffle-conditions cl:style-warning))
  
  `(progn
     (export ',name)
     ;; You can use this macro for testing other backends, other dtypes.
     (defmacro ,name (&rest backend)
       #+sbcl(declare (sb-ext:muffle-conditions cl:style-warning))
       `(let ((*using-backend* (append *using-backend* ',@backend)))
	  (deftest ,(symb 'case- ',name)
	    ,@(map
	       'list
	       #'(lambda (dtype)
		   `(testing (format nil "[~a]" ',dtype)
		      (ok
		       (with-dtype ,dtype
			 (progn
			   (let ((result (progn ,,@body)))
			     (if (eql result t)
				 t
				 (if (eql result :backward)
				     (fail "backward")
				     (fail "forward")))))))))
	       `,,(case op-type
		    (:sparse `(list ,@*sparse-types*))
		    (:dense  `(list ,@*dense-types*))
		   (:all `(list ,@*sparse-types* ,@*dense-types*)))))))))

(defun M= (tensor1 tensor2)
  (every #'= (tensor-vec tensor1) (tensor-vec tensor2)))

(defun ~= (x y)
  (< (- x y) 0.00001))

