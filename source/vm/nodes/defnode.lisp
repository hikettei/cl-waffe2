
(in-package :cl-waffe2/vm.nodes)

(defpackage :cl-waffe2/vm.nodes.facets-tmp)

(defparameter *facet-monopoly-mode* nil "If t, only use devices with Priority1, otherwise an error will occur.")

(defun list-of-abstracttensor-p (list)
  "Return t if LIST is non nil and contains only AbstractTensor."
  (and (consp list)
       (every #'(lambda (x) (subtypep x 'cl-waffe2/vm.generic-tensor:AbstractTensor)) list)))

(deftype list-of-abstracttensor ()
  `(and list (satisfies list-of-abstracttensor-p)))

(defmacro with-devices ((&rest backend-priority) &body body)
  "Through the macro with-devices, the facet of nodes are declared.
backend-priority is described as: (Priority1 Priority2 ...)"
  `(let ((*using-backend* ',backend-priority))
     (declare (type list *using-backend*))
     (mapc
      #'(lambda (x)
	  (unless (subtypep x 'cl-waffe2/vm.generic-tensor:AbstractTensor)
	    (warn "~a is not a subtype of cl-waffe2/vm.generic-tensor:AbstractTensor. If you want to extend device, extend this class first."
		  x)))
      *using-backend*)
     ,@body))

(defmacro with-single-device ((device-name) &body body)
  "Under this macro, cl-waffe only use devices with device-name, otherwise an error will occur"
  `(let ((*facet-monopoly-mode* t))
     (with-backend (,device-name)
       ,@body)))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun symb (&rest inputs)
    (intern (with-output-to-string (out) (dolist (sym inputs) (princ sym out)))))
  (defun subnode-name (abstract-node device)
    (intern (with-output-to-string (out)
	      (princ abstract-node out)
	      (princ '- out)
	      (princ device out))
	    'cl-waffe2/vm.nodes.facets-tmp)))

(defun determine-facet-of-nodes (abstract-name devices)
  (declare (type list devices)
	   (type symbol abstract-name))
  (loop for device in devices
	do (let ((node-name (subnode-name abstract-name device)))
	     (when (subtypep node-name abstract-name)
	       (return-from determine-facet-of-nodes
		 node-name))
	     
	     (when *facet-monopoly-mode*
	       (error "Not Found Case1 (Debug Mode)"))))
  
  (error "Couldn't find the node"))

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

  (let ((initarg-slots (map 'list #'(lambda (slots)
				      ;; Auto-Generated Constructor is Enabled Only When:
				      ;; slot has :initarg
				      ;; slot-name corresponds with any of constructor-arguments
				      (when
					  (and
					   (find (first slots) (flatten constructor-arguments))
					   (find :initarg slots))
					slots))
			    slots)))
    (flet ((parse-initarg-slot (slot)
	     (when slot
	       ;; constraints: slot has :initarg
	       `(list ,(intern (symbol-name (nth (1+ (position :initarg slot)) slot)) "KEYWORD")
		      ,(car slot)))))
      `(eval-when (:compile-toplevel :load-toplevel :execute)
	   (defclass ,abstract-name (AbstractNode)
	     (,@slots)
	     (:documentation ,documentation))
	 ;; Backends are modular
	 (defun ,abstract-name (,@(cdr constructor-arguments))
	   ,documentation
	   (let* ((,subscript-p (create-subscript-p ,where))
		  (,(car constructor-arguments)
		    (apply #'make-instance (determine-facet-of-nodes ',abstract-name *using-backend*)
				   (list :function-node ,subscript-p)
				   ,@(map 'list #'parse-initarg-slot initarg-slots))))
	     (declare (ignorable ,(car constructor-arguments)))
	     ,@constructor-body
	     ;; Backendに応じてNodeのsubclassを生成
	     (the ,abstract-name ,(car constructor-arguments))))))))


(defmacro define-impl ((abstract-name
			&key
			  (device))
		       &key
			 forward
			 backward
		       &aux
			 (inputs (gensym "inputs")))
  "Through the macro define-impl, the behaviour of nodes are described.

Follow these constraints:

1. Arguments must be this format:
   Forward  -> (node input-tensor1 input-tensor2 ...)
   Backward -> (node dy)

   Other parameters should be given as constructor."
  (let ((forward-self-name (caar forward))
	(backward-self-name (caar backward))
	(forward-args  (cdar forward))
	(backward-args (cdar backward))
	(forward-body  (cdr forward))
	(backward-body (cdr backward))
	(impl-name (subnode-name abstract-name device)))

    (eval-when (:compile-toplevel :load-toplevel :execute)
      (assert (subtypep device 'cl-waffe2/vm.generic-tensor:AbstractTensor)
	      nil
	      "Assetion Failed because the node ~a 's :device (~a) is not subtype of cl-waffe2/vm.generic-tensor:AbstractTensor."
	      abstract-name
	      device)

      (assert (= (length backward-args) 1)
	      nil
	      "Assertion Failed because the arguments of backward, must be: (node dy) but got ~a. At ~a node"
	      backward-args
	      abstract-name))

    `(progn
       (defclass ,impl-name (,abstract-name)
	 nil
	 (:documentation ,(format nil "The node ~a is a one facet of ~a for the device ~a. Automatically defined by cl-waffe."
				  impl-name
				  abstract-name
				  device)))
       ;; TODO: Auto generate of documentations
       (defmethod forward ((,forward-self-name ,impl-name) &rest ,inputs)
	 (declare (type ,impl-name ,forward-self-name))
	 (multiple-value-bind (,@forward-args) (apply #'values ,inputs)
	   (declare (type ,device ,@forward-args))
	   ,@forward-body))
       (defmethod backward ((,backward-self-name ,impl-name) ,@backward-args)
	 (declare (type ,impl-name ,backward-self-name)
		  (type ,device ,@backward-args))
	 ,@backward-body))))

;; Tests
;; backward-test-toolみたいなのが欲しい
;; 可視化のためのツールを頑張る
;; 次にやる：Define-impl
;; Macro-Basedの遅延評価
;; Backwardの構築
;; Shape評価
;; Tensorに計算ノードを構築する
;; (build tensor)みたいなノードで(works like pytorch's a.backward())計算ノードを受理 逆伝播の関数順伝播の関数を構築
;; (value tensor)で今までのノードを動的に構築 (cl-waffeと同じ)
;; どこでvalueしてもok. debug is ez.
;; (build tensor)で関数構築するとき、Lispのコードも残しておく (i.e.: print debugなどもしやすい)

;; Constaints 引数は以下の形式でないといけない
;; Forward: (Node device-tensor1 device-tensor2 ...)
;; Backward: (Node device-tensor1)

;; TODO: Generic Where
;; RNNみたいに(values x hs)を返したいときはどうする？
;; outのコードは一パターンのみ
;; (nth hoge result)を挟む...
;; ViewNode

(defnode (AddNode (myself)
	  :where `([x y] [x y] -> [x y])
	  :documentation "x + y"))

(defnode (MulNode (myself)
	  :where `([x y] [x y] -> [x y])
	  :documentation "x * y"))


(define-impl (AddNode :device cl-waffe2/vm.generic-tensor:CPUTensor)
	     :forward ((self x y)
		       `(+ (tensor-vec ,x)
			   (tensor-vec ,y)))
	     :backward ((self dy)
			`(values ,dy ,dy)))

(define-impl (MulNode :device cl-waffe2/vm.generic-tensor:CPUTensor)
	     :forward ((self x y)
		       `(* (tensor-vec ,x) (tensor-vec ,y)))
	     :backward ((self dy)
			`(values ,dy ,dy)))

(defun build-test ()
  (let ((x (make-tensor `(10 10)))
	(y (make-tensor `(10 10)))
	(z (make-tensor `(10 10))))
    (setf (tensor-vec x) 10)
    (setf (tensor-vec y) 5)
    (setf (tensor-vec z) 2)
    
    (forward (AddNode) (forward (MulNode) x y) z)))
