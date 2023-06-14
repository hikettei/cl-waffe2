
(in-package :cl-user)

(defpackage :cl-waffe2/base-impl.test
  (:use :cl
        :fiveam
        :cl-waffe2
	:cl-waffe2/base-impl
        :cl-waffe2/vm.generic-tensor
        :cl-waffe2/vm.nodes))

(in-package :cl-waffe2/base-impl.test)

(def-suite :base-impl-test)
(in-suite :base-impl-test)

(defparameter *dense-types*  `(:float :double))
(defparameter *sparse-types* `(:int8 :int16 :int32))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun symb (&rest inputs)
    (intern (with-output-to-string (out) (dolist (sym inputs) (princ sym out))))))

(defmacro define-tester (name op-type &body body)
  (declare (type (member :dense :sparse :all) op-type))
  `(eval-when (:compile-toplevel :load-toplevel :execute)
     (export ',name)
     ;; You can use this macro for testing other backends, other dtypes.
     
     (defmacro ,name (backend)
       `(let ((*using-backend* '`(,,backend)))
	  ,@(map 'list #'(lambda (dtype)
			   `(test ,(symb ', name '- dtype '- backend)
			      (is (with-dtype ,dtype
				    (let ((result (progn ,,@body)))
				      (if (eql result t)
					  t
					  (if (eql result :backward)
					      (error "the result of backward is invaild")
					      (error "the result of forward is invaild"))))))))
		 ',(case op-type
		     (:sparse *sparse-types*)
		     (:dense  *dense-types*)
		     (:all `(,@*sparse-types* ,@*dense-types*))))))))

