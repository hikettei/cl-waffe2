
(in-package :cl-waffe2/nn)

;;
;; 
;; 
;; 
;;

(defun conv-out-size (in padding dilation kernel-size stride)
  (floor
   (+ 1 (/ (+ in
	      (* 2 padding)
	      (* (- dilation)
		 (- kernel-size 1))
	      -1)
	   stride))))

(defun pool-out-size (in padding kernel-size stride)
  (floor
   (+ 1 (/ (+ in (* 2 padding) (- kernel-size)) stride))))

(defmodel (MaxPool2D (self kernel-size
			   &key
			   (stride kernel-size)
			   (padding 0)
			   &aux
			   (stride (maybe-tuple stride 'stride))
			   (padding (maybe-tuple padding 'padding)))
	   :slots ((kernel-size :initarg :kernel-size :type list)
		   (stride :initarg :stride)
		   (padding :initarg :padding))
	   :documentation "
Applies a 2D max pooling over an input signal composed of several input planes.

### Inputs

`kernel-size[list]` the size of window

`stride[fixnum or list]` the stride of window

`padding[fixnum or list]` adds 0 padding

Likewise `Conv2D`, these parameters can be set for both X and Y axis directions.
"
	   ;; [TODO]: Delete (if (numberp ...))
	   :where (Input[N C H_in W_in] -> Output[N C H_out W_out]
			   where
			   H_out = (if (numberp H_in)
				       (pool-out-size H_in (car padding) (car kernel-size) (car stride))
				       -1)
			   W_out = (if (numberp W_out)
				       (pool-out-size W_in (second padding) (second kernel-size) (second stride))
				       -1))
	   :on-call-> apply-maxpool2d))

(defmodel (AvgPool2D (self kernel-size
			   &key
			   (stride kernel-size)
			   (padding 0)
			   &aux
			   (stride (maybe-tuple stride 'stride))
			   (padding (maybe-tuple padding 'padding)))
	   :slots ((kernel-size :initarg :kernel-size :type list)
		   (stride :initarg :stride)
		   (padding :initarg :padding))
	   :documentation "
Applies a 2D average pooling over an input signal composed of several input planes.

### Inputs

`kernel-size[list]` the size of window

`stride[fixnum or list]` the stride of window

`padding[fixnum or list]` adds 0 padding

Likewise `Conv2D`, these parameters can be set for both X and Y axis directions.
"
	   ;; [TODO]: Delete (if (numberp ...))
	   :where (Input[N C H_in W_in] -> Output[N C H_out W_out]
			   where
			   H_out = (if (numberp H_in)
				       (pool-out-size H_in (car padding)    (car kernel-size) (car stride))
				       -1)
			   W_out = (if (numberp W_out)
				       (pool-out-size W_in (second padding) (second kernel-size) (second stride))
				       -1))
	   :on-call-> apply-avgpool2d))

(defmethod apply-maxpool2d ((self MaxPool2D) input)
  (with-slots ((stride stride) (kernel-size kernel-size) (padding padding)) self
    (multiple-value-bind (N C H-in W-in) (apply #'values (shape input))
      (let ((H-out (pool-out-size H-in (first padding)  (first kernel-size)  (first stride)))
	    (W-out (pool-out-size W-in (second padding) (second kernel-size) (second stride))))
	(call-> input
		(asnode #'unfold  `(1 1) kernel-size stride padding)
		(asnode #'!reshape t (apply #'* kernel-size))
		(asnode #'!max     :axis 1)
		(asnode #'!reshape N H-out W-out C)
		(asnode #'!permute 3 0 2 1))))))

(defmethod apply-avgpool2d ((self AvgPool2D) input)
  (with-slots ((stride stride) (kernel-size kernel-size) (padding padding)) self
    (multiple-value-bind (N C H-in W-in) (apply #'values (shape input))
      (let ((H-out (pool-out-size H-in (first padding)  (first kernel-size)  (first stride)))
	    (W-out (pool-out-size W-in (second padding) (second kernel-size) (second stride))))
	(call-> input
		(asnode #'unfold  `(1 1) kernel-size stride padding)
		(asnode #'!reshape t (apply #'* kernel-size))
		(asnode #'!mean     :axis 1)
		(asnode #'!reshape N H-out W-out C)
		(asnode #'!permute 3 0 2 1))))))

