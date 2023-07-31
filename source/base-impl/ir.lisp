
(in-package :cl-waffe2/base-impl)

;; ir.lisp provides AbstractNodes and its implementation of controling flows of nodes

;; Principle Nodes:
;;   IfNode
;;   LoopNode


;; Unsafe of number of arguments
;; There should be something better way to deal with flexible arguments.

;; IfNode ... then/elseの帰り値が複数だったら？？？

#|
(defnode (IfNode (self condition then else)
	  :documentation "
```lisp
             out
              |
   [IfNode cond = `(> a 1)]
 then  |              | else
    [Node1]         [Node2]
       |              |
       ----------------
              |
           result
```

(call (IfNode `(> n 1) (SinNode) (CosNode))
      (randn `(3 3))
      (randn `(3 3)))
"
	  :where (Out[~] -> Result[~]) ;; Multiple arguments?
	  :slots ((condition :initarg :condition :reader cond-of :type list)
		  (then :initarg :then :reader then-of :type AbstractTensor)
		  (else :initarg :else :reader else-of :type AbstractTensor)
		  (built-then :accessor then-model :type Compiled-Composite)
		  (built-else :accessor else-model :type Compiled-Composite))
	  :backward ((self dout out)
		     (declare (ignore dout out))
		     (let ((then (then-model self))
			   (else (else-model self)))
		       (values
			(!mul 0 (lazy-if (cond-of self)
					 then
					 else)))))))


(define-impl (IfNode :device t
		     :cache-when-compiled nil)
	     :forward ((self out)
		       (declare (ignore out))
		       (let ((then (build (then-of self)))
			     (else (build (else-of self))))
			 (setf (then-model self) then
			       (else-model self) else)
			 `(progn
			    (cl-waffe2/vm.generic-tensor::declare-compiled-composite ,then)
			    (cl-waffe2/vm.generic-tensor::declare-compiled-composite ,else)
			    (if ,(cond-of self)
				(forward ,then)
				(forward ,else))))))

(defmacro lazy-if (condition then else)
  `(call (IfNode ,condition ,then ,else) (make-clone ,then)))


(defun ?all (tensor)
  (print tensor)
  nil)

(defun test-case ()
  (let ((a (!sin (ax+b `(3 3) 1 0)))
	(b (!sin (ax+b `(3 3) 1 0))))
    (lazy-if (?all (A>B a b))
	     (lazy-print (!sum (!sub a b)))
	     (lazy-print (!sum (!add a b))))))

|#


