
(in-package :cl-waffe2/vm.generic-tensor)

;; Here we provide two macros:
;; define-tensor (Tensors and Backend are strongly combined.)
;; CFFI-Style
;; Column-Major And Row-Major

(defun order-p (name)
  (declare (type keyword name))
  (or (eql name :column) (eql name :row)))

;; cl-xmatrixからView関連のコードを移植しておく

(defclass AbstractTensor ()
  ((nodes :initarg :nodes :initform nil :reader tensor-nodes :type list)
   (orig-shape :initarg :shape :initform nil :reader original-shape :type list)
   (visible-shape :initarg :visible-shape :initform nil :reader shape :type list)
   (view :initarg :view :initform nil :reader view :type list)
   (previous-state :initform nil :reader tensor-state)
   (requires-grad :initform nil :initarg :requires-grad :type boolean)
   (order :initform :column :initarg :order :type (satisfies order-p))
   (trace-state :initform nil)))



(defmacro assure-dimensions (mat1 mat2)
  "Do nothing if mat1 and mat2 are the same shape, otherwise throw shaping-error."
  `(if (equal (the list (shape ,mat1)) (the list (shape ,mat2)))
	 t
	 (shaping-error "Two matrices: ~a and ~a couldn't operated together." (shape ,mat1) (shape ,mat2))))
