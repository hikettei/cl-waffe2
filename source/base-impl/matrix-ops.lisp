
(in-package :cl-waffe2/base-impl)

;; Implement: Matmul/Dot/ArgMax/ArgMin

(defnode (MatMulNode (myself &key transpose-a transpose-b)
	  :where `(A[~ i j] B[~ j k] C[~ i k] -> C[~ i k])
	  :slots ((transpose-a :initarg :transpose-a :type boolean :reader trans-a?)
		  (transpose-b :initarg :transpose-b :type boolean :reader trans-b?))
	  :backward ((self dout da db do)
		     (declare (ignore do))
		     (values
		      (!matmul dout db :transpose-y (not (trans-b? self)))
		      (!matmul da dout :transpose-x (not (trans-a? self)))
		      nil))
	  :documentation ""))

(defnode (LazyTransposeNode (myself)
	  :where `(A[~ i j] -> A[~ j i])
	  :documentation "LazyTransposeNode is the matmul-dedicated node which supplies the lazy-transpose feature.

Internally, This Node Returns The Given A itself but taking transpose of A's shape.

If the computation node is like: [LazyTransposeNode] -> [MatmulNode], then transpose will be done with NO overhead."))

(define-impl (LazyTransposeNode)
	     :forward ((self x)
		       `(progn ,x))
	     :backward ((self dout dx)
			(declare (ignore dx))
			(values dout)))

(defun transposed-p (tensor)
  "Return T if previous-node is LazyTransposeNode"
  (subtypep (class-of (tensor-backward tensor)) 'LazyTransposeNode))

(defun !t (tensor)
  "Applies Lazy-Transpose to the given tensor"
  (forward (LazyTransposeNode) tensor))


;; On Backward or when transposed, Maybe The Result become 0.0 (BUG)
(defun !matmul (x y
		&key
		  (out nil)
		  (transpose-x nil)
		  (transpose-y nil)
		&aux
		  (transpose-x (if (transposed-p x)
				   nil
				   transpose-x))
		  (transpose-y (if (transposed-p y)
				   nil
				   transpose-y)))
  "X[~ i j] @ Y[~ j k] -> C[~ i k]"
  (let* ((i  (nth (if transpose-x 1 0) (last (shape x) 2)))
	 (jx (nth (if transpose-x 0 1) (last (shape x) 2)))
	 (jy (nth (if transpose-y 1 0) (last (shape y) 2)))
	 (k  (nth (if transpose-y 0 1) (last (shape y) 2)))
	 (out (or out (make-input `(,@(butlast (shape x) 2) ,i ,k) nil
				  :dtype (dtype x)
				  :order (order x)))))

    ;; TODO: More Detailed Errors.
    (assert (= jx jy) nil "Assertion Failed with X[~~ i j] @ Y[~~ j k] -> C[~~ i k]. ~a and ~a" (shape x) (shape y))

    (forward (MatmulNode :transpose-a transpose-x
			 :transpose-b transpose-y)
	     (if transpose-x
		 (!t x)
		 x)
	     (if transpose-y
		 (!t y)
		 y)
	     out)))

;; (defun !dot (a b))

;; (defun !argmax)
;; (defun !argmin)
;; (defun !max)
;; (defun !min)
