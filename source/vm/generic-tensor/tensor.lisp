
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
   (stride :initform nil :reader tensor-stride :type list)
   (visible-shape :initform nil :reader shape :type list)
   (view :initarg :view :initform nil :reader view :type list)
   (projected-p :initarg :projected-p :initform nil :type boolean :reader tensor-projected-p)
   ;; (vec)
   (previous-state :initform nil :reader tensor-state)
   (requires-grad :initform nil :initarg :requires-grad :type boolean)
   (order :initarg :order :initform :column :type (satisfies order-p) :accessor order)
   (trace-state :initform nil)))

(defmethod initialize-instance :after ((tensor AbstractTensor) &rest initargs &key &allow-other-keys)
  (declare (ignore initargs))
  (with-slots ((stride stride) (order order) (visible-shape visible-shape) (view view)) tensor
    ;; visible area
    (setf stride (calc-strides tensor order))
    (setf visible-shape (parse-view-subscripts tensor (or nil `(t))))
    nil))

(defmacro assure-dimensions (mat1 mat2)
  "Does nothing if mat1 and mat2 has a completely the same shape, otherwise throws shaping-error."
  `(if (equal (the list (shape ,mat1)) (the list (shape ,mat2)))
       t
       (shaping-error "Assertion Failed because two matrices ~a and ~a couldn't operated together." (shape ,mat1) (shape ,mat2))))

(defmethod calc-strides ((tensor AbstractTensor) (order (eql :column)))
  "Computes column-major-strides"
  (column-major-calc-strides (slot-value tensor 'orig-shape)))

(defmethod calc-strides ((tensor AbstractTensor) (order (eql :row)))
  "Computes row-major-strides"
  (row-major-calc-strides (slot-value tensor 'orig-shape)))

;; Tensors must support displace-to

;; (defmethod print-object ((tensor AbstractTensor) stream))

