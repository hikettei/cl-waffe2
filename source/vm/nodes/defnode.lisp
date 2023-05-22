
(in-package :cl-waffe2/vm.nodes)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun symb (&rest inputs)
    (intern (with-output-to-string (out) (dolist (sym inputs) (princ sym out))))))

(defmacro defnode ((abstract-name
		   (&rest constructor-arguments)
		    &key
		      (where t)
		      (slots nil)
		      (documentation ""))
		   &body constructor-body
		     &aux (subscript-p (gensym)))
  "The macro defnode helps you to describe how nodes are working.

abstract-name... The Common Name for your node.

Common Name
 |-> Sub Node(:backend = :cpu)
  ...

=================================================================
Shaping Format:

Ignore with t.

"

  `(prog1
       (defclass ,abstract-name (AbstractNode)
	 ,slots
	 (:documentation ,documentation))
     
     (defun ,abstract-name (,@(cdr constructor-arguments))
       (let* ((,subscript-p (create-subscript-p ,where))
	      (,(car constructor-arguments)
		(make-instance ',abstract-name
			       :function-node ,subscript-p)))
	 (declare (ignorable ,(car constructor-arguments)))
	 ,@constructor-body
	 ;; Backendに応じてNodeのsubclassを生成
	 ,(car constructor-arguments)))))

(defmacro define-impl ((abstract-name
			&key
			  (device))
		       &key
			 forward
			 backward
			 &aux (inputs (gensym "inputs")))
  "Adds Backend"
  (let ((forward-self-name (caar forward))
	(backward-self-name (caar backward))
	(forward-args  (cdar forward))
	(backward-args (cdar backward))
	(forward-body  (cdr forward))
	(backward-body (cdr backward))
	(impl-name (symb abstract-name device)))
    ;; assert (length backward-args) == 1
    `(prog1
	 (defclass ,impl-name (,abstract-name)
	   nil
	   (:documentation ,(format nil "Automatically defined by cl-waffe")))
       (defmethod forward ((,forward-self-name ,impl-name) &rest ,inputs)
	 ;; Error Check
	 (multiple-value-bind (,@forward-args) (apply #'values ,inputs)
	   (let ((,forward-self-name ,forward-self-name))
	     ,@forward-body)))
       (defmethod backward ((,backward-self-name ,impl-name) ,@backward-args)
	 ,@backward-body))))

;; Tests
(define-impl (AddNode :device CPUTensor)
	     :forward ((node x y)
		       (declare (ignore node))
		       (+ x y))
	     :backward ((node dy)
			))

(defnode (AddNode (myself)
	  :where `([~] [~] -> [~])
	  :documentation "The Node Addnode Provides ..."))
