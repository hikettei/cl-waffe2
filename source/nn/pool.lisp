
(in-package :cl-waffe2/nn)

;;
;; 
;; 
;; 
;;

;; TODO: Bugfix of !permute

(defun conv-out-size (in padding dilation kernel-size stride)
  (floor
   (+ 1 (/ (+ in
	      (* 2 padding)
	      (* (- dilation)
		 (- kernel-size 1))
	      -1)
	   stride))))

(defmodel (MaxPool2D (self kernel-size
			   &key
			   (stride 1)
			   (padding 0)
			   (dilation 1)
			   &aux
			   (stride (maybe-tuple stride 'stride))
			   (padding (maybe-tuple padding 'padding))
			   (dilation (maybe-tuple dilation 'dilation)))
	   :slots ((kernel-size :initarg :kernel-size :type list)
		   (stride :initarg :stride)
		   (padding :initarg :padding)
		   (dilation :initarg :dilation))
	   :documentation "
Applies a 2D max pooling over an input signal composed of several input planes.

### Inputs

`kernel-size[list]` the size of window

`stride[fixnum or list]` the stride of window

`padding[fixnum or list]` adds 0 padding

`dilation[fixnum or list]` a parameter that controls the stride of elements in the window.

Likewise `Conv2D`, these parameters can be set for both X and Y axis directions.
"
	   ;; [TODO]: Delete (if (numberp ...))
	   :where (Input[N C H_in W_in] -> Output[N C H_out W_out]
			   where
			   H_out = (if (numberp H_in)
				       (conv-out-size H_in (car padding) (car dilation) (car kernel-size) (car stride))
				       -1)
			   W_out = (if (numberp W_out)
				       (conv-out-size W_in (second padding) (second dilation) (second kernel-size) (second stride))
				       -1))
	   :on-call-> apply-maxpool2d))

(defmodel (AvgPool2D (self kernel-size
			   &key
			   (stride 1)
			   (padding 0)
			   (dilation 1)
			   &aux
			   (stride (maybe-tuple stride 'stride))
			   (padding (maybe-tuple padding 'padding))
			   (dilation (maybe-tuple dilation 'dilation)))
	   :slots ((kernel-size :initarg :kernel-size :type list)
		   (stride :initarg :stride)
		   (padding :initarg :padding)
		   (dilation :initarg :dilation))
	   :documentation "
Applies a 2D average pooling over an input signal composed of several input planes.

### Inputs

`kernel-size[list]` the size of window

`stride[fixnum or list]` the stride of window

`padding[fixnum or list]` adds 0 padding

`dilation[fixnum or list]` a parameter that controls the stride of elements in the window.

Likewise `Conv2D`, these parameters can be set for both X and Y axis directions.
"
	   ;; [TODO]: Delete (if (numberp ...))
	   :where (Input[N C H_in W_in] -> Output[N C H_out W_out]
			   where
			   H_out = (if (numberp H_in)
				       (conv-out-size H_in (car padding) (car dilation) (car kernel-size) (car stride))
				       -1)
			   W_out = (if (numberp W_out)
				       (conv-out-size W_in (second padding) (second dilation) (second kernel-size) (second stride))
				       -1))
	   :on-call-> apply-avgpool2d))

(defmethod apply-maxpool2d ((self MaxPool2D) input)
  (with-slots ((stride stride) (kernel-size kernel-size) (padding padding) (dilation dilation)) self
    (multiple-value-bind (N C H-in W-in) (apply #'values (shape input))
      (let ((H-out (conv-out-size H-in (car padding) (car dilation) (car kernel-size) (car stride)))
	    (W-out (conv-out-size W-in (second padding) (second dilation) (second kernel-size) (second stride)))
	    (p-y (mod
		  (-
		   (second stride)
		   (mod
		    (+ h-in (* 2 (second padding))
		       (- (* (second dilation) (- (second kernel-size) 1))))
		    (second stride)))
		  (second stride))) ;; (sY - (iH + pY * 2 - (kH - 1) * dY) % sY) % sY
	    (p-x (mod
		  (-
		   (car stride)
		   (mod
		    (+ w-in (* 2 (car padding))
		       (- (* (car dilation) (- (car kernel-size) 1))))
		    (car stride)))
		  (car stride)))) ;; (sX - (iW + pX * 2 - (kW - 1) * dX) % sX) % s

	(call-> input
		(asnode #'padding    `(t t (,(second padding) ,(+ (second padding) p-y)) (,(car padding) ,(+ (car padding) p-x))))
		(asnode #'!im2col-cpu N C (second kernel-size) (car kernel-size) h-out w-out (car stride) (second stride))
		(asnode #'!reshape    t (apply #'* kernel-size))
		(asnode #'!max        :axis 1)
		(asnode #'!reshape    N h-out w-out C)
		(asnode #'!permute    3 0 1 2))))))


(defmethod apply-avgpool2d ((self AvgPool2D) input)
  (with-slots ((stride stride) (kernel-size kernel-size) (padding padding) (dilation dilation)) self
    (multiple-value-bind (N C H-in W-in) (apply #'values (shape input))
      (let ((H-out (conv-out-size H-in (car padding) (car dilation) (car kernel-size) (car stride)))
	    (W-out (conv-out-size W-in (second padding) (second dilation) (second kernel-size) (second stride)))
	    (p-y (mod
		  (-
		   (second stride)
		   (mod
		    (+ h-in (* 2 (second padding))
		       (- (* (second dilation) (- (second kernel-size) 1))))
		    (second stride)))
		  (second stride))) ;; (sY - (iH + pY * 2 - (kH - 1) * dY) % sY) % sY
	    (p-x (mod
		  (-
		   (car stride)
		   (mod
		    (+ w-in (* 2 (car padding))
		       (- (* (car dilation) (- (car kernel-size) 1))))
		    (car stride)))
		  (car stride)))) ;; (sX - (iW + pX * 2 - (kW - 1) * dX) % sX) % s

	(call-> input
		(asnode #'padding    `(t t (,(second padding) ,(+ (second padding) p-y)) (,(car padding) ,(+ (car padding) p-x))))
		(asnode #'!im2col-cpu N C (second kernel-size) (car kernel-size) h-out w-out (car stride) (second stride))
		(asnode #'!reshape    t (apply #'* kernel-size))
		(asnode #'!mean       :axis 1)
		(asnode #'!reshape    N h-out w-out C)
		(asnode #'!permute    3 0 1 2))))))

