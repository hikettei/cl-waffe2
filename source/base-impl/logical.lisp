
(in-package :cl-waffe2/base-impl)

;; =============================================
;; Defnode Parts
;; =============================================


(export '(logical-condition logical-true-then logical-false-then
	  Where-Operation-Node Compare-Operation-Node))

(defnode (Where-Operation-Node (myself condition true-then false-then)
	  ;;:no-grad t
	  :where (A[~] OUT[~] -> OUT[~])
	  :slots ((condition :initarg :condition :type function :reader logical-condition)
		  (true-then :initarg :true-then :reader logical-true-then)
		  (false-then :initarg :false-then :reader logical-false-then))
	  :backward ((self dout da do)
		     (declare (ignore dout da do))
		     (values nil nil))))

(defnode (Compare-Operation-Node (myself condition true-then false-then)
	  ;;:no-grad t
	  :where (A[~] B[~] OUT[~] -> OUT[~])
	  :slots ((condition :initarg :condition :type function :reader logical-condition)
		  (true-then :initarg :true-then :reader logical-true-then)
		  (false-then :initarg :false-then :reader logical-false-then))
	  :backward ((self dout da db do)
		     (declare (ignore dout da db do))
		     (values nil nil nil))))


;; =============================================
;; Fundamental APIs
;; =============================================

;; how to determine dtype
;; can compare being optimized?
(export '!where)
(defun !where (tensor condition &key (true-then 1) (false-then 0) (out nil))
  ""
  (declare (type AbstractTensor tensor)
	   (type function condition)
	   (type number true-then false-then))

  (assert (not (scalar-p tensor))
	  nil
	  "!where: Assertion Failed with the given tensor isn't ScalarTensor.")

  (let ((out (or out (make-input (shape tensor) nil
				 :order (order tensor)
				 :dtype (dtype tensor))))
	(true-then  (coerce true-then (dtype->lisp-type (dtype tensor))))
	(false-then (coerce false-then (dtype->lisp-type (dtype tensor)))))
    (forward (Where-Operation-Node condition true-then false-then)
	     tensor
	     out)))

(export '!compare)
(defun !compare (tensor1 tensor2 condition &key (true-then 1) (false-then 0) (out nil))
  ""
  (declare (type AbstractTensor tensor1 tensor2)
	   (type function condition)
	   (type number true-then false-then))

  (assert (and (not (scalar-p tensor1)) (not (scalar-p tensor2)))
	  nil
	  "!compare: Assertion Failed because the given tensor1, and tensor2 should be a matrix, not a scalar.")

  (let ((out (or out (make-input (shape tensor1) nil
				 :order (order tensor1)
				 :dtype (dtype tensor1))))
	(true-then  (coerce true-then  (dtype->lisp-type (dtype tensor1))))
	(false-then (coerce false-then (dtype->lisp-type (dtype tensor1)))))
    (forward (Compare-Operation-Node condition true-then false-then)
	     tensor1
	     tensor2
	     out)))

;; =============================================
;; More APIs
;; =============================================

(macrolet ((define-cmp-scal-operation (name operator)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (A scal &key (out nil) (true-then 1) (false-then 0))
		  ""
		  (!where A #'(lambda (x) (,operator x scal))
			  :true-then  true-then
			  :false-then false-then
			  :out out)))))
  (define-cmp-scal-operation A>scal >)
  (define-cmp-scal-operation A<scal <)
  (define-cmp-scal-operation A>=scal >=)
  (define-cmp-scal-operation A<=scal <=))

(macrolet ((define-cmp-mat-operation (name operator)
	     `(eval-when (:compile-toplevel :load-toplevel :execute)
		(export ',name)
		(defun ,name (A B &key (out nil) (true-then 1) (false-then 0))
		  ""
		  (!compare A B #',operator
			    :true-then  true-then
			    :false-then false-then
			    :out out)))))
  (define-cmp-mat-operation A>B >)
  (define-cmp-mat-operation A<B <)
  (define-cmp-mat-operation A>=B >=)
  (define-cmp-mat-operation A<=B <=))
		       

