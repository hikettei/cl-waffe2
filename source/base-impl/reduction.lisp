
(in-package :cl-waffe2/base-impl)


;;========================================================================
;; !sum semantics:
;; Let A be a 3 x 3 Tensor, the operation is to sum up A along axis=1.
;; 1. Prepare A
;; +++
;; +++
;; +++
;; 2. Prepare A-out 3 * 1
;; +++
;; 3. Broadcast A-out along axis=1
;; +++
;; ---
;; ---
;; 4. Adds A and A-out element-wise.
;; This could be applied whenever the given axis is consisted of axes of list.
;;=========================================================================

(defun !sum (tensor &key (axis t) (-> nil) (keep-repeat nil))
  "Sum up the given tensor along the axis."
  (declare (type AbstractTensor tensor)
	   (type boolean keep-repeat)
	   (type (or t list fixnum) axis))
  (let* ((shape (copy-list (shape tensor)))
	 (view-args (make-list (length shape) :initial-element t))
	 (dims  (length shape)))

    ;; Compute Reduction Size.
    ;; Parse -1 -> 1 for example.
    (typecase axis
      (fixnum
       (if (< axis 0)
	   (setq axis (+ axis dims)))

       (setf (nth axis view-args) `(:broadcast ,(nth axis shape)))
       (setf (nth axis shape) 1))
      (list
       (dolist (dim axis)
	 (let ((tgt (if (< dim 0)
			(+ dim dims)
			dim)))
	   (setf (nth tgt view-args) `(:broadcast ,(nth tgt shape)))
	   (setf (nth tgt shape) 1))))
      (T
       (setq view-args (loop for s in shape
			     collect `(:broadcast ,s)))
       (setq shape (make-list dims :initial-element 1))))

    ;; Use Instead: make-input
    (let* ((out (or -> (make-tensor shape
				    :dtype (dtype tensor)
				    :order (order tensor))))
	   (out (A*=scal out 0))) ;; TODO: !mul is nothing but extravagance to fill with 0.0!, replace this op with !fill

      (assert (equal (shape out) shape)
	      nil
	      "!sum: Assertion Failed because the given out's shape is ~a, but excepted: ~a" (shape out) shape)

      ;; Main Parts
      (multiple-value-bind (out* reverser) (apply #'!view out view-args)
	(if keep-repeat
	    (A+=B out* tensor)
	    (apply #'!view (A+=B out* tensor) reverser))))))

(defun !mean (tensor &key (axis t) (-> nil) (keep-repeat nil))
  ""
  (let* ((result (!sum tensor :axis axis :-> -> :keep-repeat keep-repeat))
	 (dims (length (shape tensor)))
	 (reducted-elements 1))
    
    (typecase axis
      (fixnum
       (if (< axis 0)
	   (setq axis (+ axis dims)))

       (setq reducted-elements (nth axis (shape tensor))))
      (list
       (dolist (dim axis)
	 (let ((tgt (if (< dim 0)
			(+ dim dims)
			dim)))
	   (setq reducted-elements (* reducted-elements (nth tgt (shape tensor)))))))
      (T (setq reducted-elements (apply #'* (shape tensor)))))

    (!scalar-div reducted-elements result)))

