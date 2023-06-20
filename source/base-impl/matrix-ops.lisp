
(in-package :cl-waffe2/base-impl)

;; Implement: Matmul/Dot/ArgMax/ArgMin

(defnode (MatMulNode (myself dtype &key transpose-a transpose-b)
	  :where (A[~ i j] B[~ j k] C[~ i k] -> C[~ i k])
	  :slots ((transpose-a :initarg :transpose-a :type boolean :reader trans-a?)
		  (transpose-b :initarg :transpose-b :type boolean :reader trans-b?)
		  (A :accessor matmul-orig-a)
		  (B :accessor matmul-orig-b))
	  ;; add slots: orig-a orig-b
	  :backward ((self dout da db do)
		     (declare (ignore do))
		     (values
		      (!matmul dout (!t db))
		      (!matmul (!t da) dout)
		      nil))
	  :documentation ""))

(defnode (LazyTransposeNode (self)
	  :where (A[~ i j] -> A[~ j i])
	  :documentation "LazyTransposeNode is the matmul-dedicated node which supplies the lazy-transpose feature.

Internally, This Node Returns The Given A itself but taking transpose of A's shape.

If the computation node is like: [LazyTransposeNode] -> [MatmulNode], then transpose will be done with NO overhead."))

(define-impl (LazyTransposeNode :device t)
	     :save-for-backward (t)
	     :forward ((self x)
		       `(progn ,x))
	     :backward ((self dout dx)
			(declare (ignore dx))
			(values dout)))

(defun transposed-p (tensor)
  "Return T if previous-node is LazyTransposeNode"
  (subtypep (class-of (tensor-backward tensor)) 'LazyTransposeNode))


;; :== The problem is that ==============:
;;  !flexible(!t(x)).is_transposed? = NIL
;;  !t(!flexible(x)).is_flexible?   = T
;; :=====================================:
(defun !t (tensor)
  "Applies Lazy-Transpose to the given tensor"
  ;;(forward (LazyTransposeNode) tensor)
  (extend-states (forward (LazyTransposeNode) tensor) tensor))

(defun !matmul (x y
		&key
		  (out nil)
		  (transpose-x nil)
		  (transpose-y nil)
		&aux
		  (x (if transpose-x (!t x) x))
		  (y (if transpose-y (!t y) y))
		  (transpose-x (transposed-p x))
		  (transpose-y (transposed-p y)))
  "X[~ i j] @ Y[~ j k] -> C[~ i k]"
  (let* ((i  (nth 0 (last (shape x) 2)))
	 (jx (nth 1 (last (shape x) 2)))
	 (jy (nth 0 (last (shape y) 2)))
	 (k  (nth 1 (last (shape y) 2)))
	 ;; the way to make out's shape
	 (larger-shape (if (> (length (shape x)) (length (shape y)))
			   (shape x)
			   (shape y)))
	 ;; the longer dim's shape is adapted.
	 (out (or out (make-input `(,@(butlast larger-shape 2) ,i ,k) nil
				  :dtype (dtype x)
				  :order (order x)))))
    
    (when (not (= jx jy))
      (shaping-error
       "!matmul failed because the last two shapes of the two given matrices are invaild.
The operation is: A[~~ i j] B[~~ j k] C[~~ i k] -> C[~~ i k]
                        ^      ^
                     j doesn't match: ~a and ~a
Shapes: A = ~a, B = ~a"
       jx
       jy
       (shape x)
       (shape y)))
    
    (forward (MatmulNode (dtype x)
	      :transpose-a transpose-x
	      :transpose-b transpose-y)
	      x
	      y
	      out))) ;; !flexible

(defun !dot (x y)
  ""
  (!sum (!mul x y)))

;; (defun einsum)

(export '(ArgMax-Node ArgMin-Node !argmax !argmin))
(defnode (ArgMax-Node (myself out-size)
	  :where (A[~] OUT[out-size] -> OUT[out-size])
	  :backward ((self dout da do)
		     (declare (ignore dout da do))
		     (values nil nil)))
  (setf (ignore-shape-error myself) t))

(defnode (ArgMin-Node (myself out-size)
	  :where (A[~] OUT[out-size] -> OUT[out-size])
	  :backward ((self dout da do)
		     (declare (ignore dout da do))
		     (values nil nil)))
  (setf (ignore-shape-error myself) t))


(defun !argmax (tensor &key (axis -1) (out nil))
  "The function !argmax computes the indices of maximum values of all elements below the **axis** dimension in the given tensor.

Input:  Tensor ( ... a b c)

Return: AbstractTensor[uint32] ( ... a 1 1) If axis=-2"
  (declare (type AbstractTensor tensor)
	   (type fixnum axis))
  (let* ((axis (if (< axis 0)
		   (+ (length (shape tensor)) axis)
		   axis))
	 (out-shape (butlast (shape tensor) axis))
	 (x   (apply #'!reshape tensor `(,@out-shape t)))
	 (out (or out (make-input `(,@out-shape 1) nil
				  :dtype :uint32
				  :order (order tensor)))))
    (forward (ArgMax-Node (shape out)) x out)))

(defun !argmin (tensor &key (axis -1) (out nil))
  "The function !argmin computes the indices of minimum values of all elements below the **axis** dimension in the given tensor.

Input:  Tensor ( ... a b c)

Return: AbstractTensor[uint32] ( ... a 1 1) If axis=-2"
  (declare (type AbstractTensor tensor)
	   (type fixnum axis))
  (let* ((axis (if (< axis 0)
		   (+ (length (shape tensor)) axis)
		   axis))
	 (out-shape (butlast (shape tensor) axis))
	 (x   (apply #'!reshape tensor `(,@out-shape t)))
	 (out (or out (make-input `(,@out-shape 1) nil
				  :dtype :uint32
				  :order (order tensor)))))
    (forward (ArgMin-Node (shape out)) x out)))

;; (defun !argmax)
;; (defun !argmin)
;; (defun !max)
;; (defun !min)
