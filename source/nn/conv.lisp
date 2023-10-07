
(in-package :cl-waffe2/nn)

(defun maybe-tuple (value name)
  (typecase value
    (number `(,value ,value))
    (list
     (if (= (length value) 2)
	 value
	 (error "Conv2D: ~a should be given as list with length = 2, but got ~a" name value)))
    (t
     (error "Conv2D: ~a should be given as fixnum or list(with length=2), but got ~a" name value))))

(defmodel (Conv2D (self in-channels out-channels kernel-size
			&key
			(stride 1)
			(padding 0)
			(dilation 1)
			(groups 1)
			(bias t)
			&aux
			(stride (maybe-tuple stride 'stride))
			(kernel-size (maybe-tuple kernel-size 'kernel-size))
			(padding (maybe-tuple padding 'padding))
			(dilation (maybe-tuple dilation 'dilation)))
	   :documentation "
Applies a 2D convolution over an input signal composed of several input planes.


### Inputs

`in-channels[fixnum]` `out-channels[fixnum]` the number of channels. For example, if the input image is RGB, `in-channels=3`.

`kernel-size[list (kernel-x kernel-y)]` controls the size of kernel (e.g.: `'(3 3)`).

`padding[fixnum or list]` controls the amount of padding applies to the coming input. pads in X and Y direction when an integer value is entered. set a list of `(pad-x pad-y)` and pads in each direction.

`stride[fixnum or list]` controls the stride for cross-correlation. As with `padding`, this parameter can be applied for each x/y axis.

`dilation[fixnum or list]` controls the connections between inputs and outputs. in_channels and out_channels must both be divisible by groups. (currently not working, please set 1.)

`bias[boolean]` Set t to use bias.

### Parameters

Let k be `(/ groups (* in-channels (apply #'* kernel-size)))`.


`weight` is a trainable parameter of `(out-channels, in-channels / groups, kernel-size[0] kernel-size[1])` sampled from `U(-sqrt(k), sqrt(k))` distribution, and can be accessed with `weight-of`.

`bias` is a trainable parameter of `(out-channels)` sampled from `U(-sqrt(k), sqrt(k))` distribution, and can be accessed with `bias-of`.

Note: When `Conv2D` is initialised, the output is displayed as -1. This is because the calculation of the output is complicated and has been omitted. Once call is invoked, the output is recorded.
"
	   :slots ((weight :accessor weight-of)
		   (bias   :accessor bias-of)
		   (stride   :initarg :stride)
		   (padding  :initarg :padding)
		   (dilation :initarg :dilation)
		   (groups   :initarg :groups)
		   (kernel-size :initarg :kernel-size))
	   ;; Memo: C_in H_in W_in ... should be determined before computing.
	   :where (Input[N C_in H_in W_in] -> Output[N C_out H_out W_out]
			   where
			   C_in  = in-channels
			   C_out = out-channels
			   ;; H_out = floor(((H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1)
			   H_out = (if (numberp H_in) ;; If H_in is a symbol, return -1 (=undetermined, later determined.)
				       (floor (+ 1 (/ (+ H_in (* 2 (car padding)) (* (- (car dilation)) (- (car kernel-size) 1)) -1)
						      (car stride))))
				       -1)
			   ;; W_out = floor(((W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1)
			   W_out = (if (numberp W_in)
				       (floor (+ 1 (/ (+ W_in (* 2 (second padding)) (* (- (second dilation)) (- (second kernel-size) 1)) -1)
						      (second stride))))
				       -1))
	   :on-call-> apply-conv2d)
		       
  (assert (typep kernel-size 'list)
	  nil
	  "Conv2D: Assertion Failed because kernel-size should be given as list. but got: ~a" kernel-size)
  (let ((k (/ groups (* in-channels (apply #'* kernel-size)))))    
    (setf (weight-of self)
	  ;; Weights are (out-channels, in-channels / groups, kernel-size[0], kernel-size[1]) tensor
	  ;; which sampled from U(-sqrt(k), sqrt(k)) dist.
	  (uniform-random
	   `(,out-channels ,(/ in-channels groups) ,(car kernel-size) ,(second kernel-size))
	   (- (sqrt k))
	   (sqrt k)
	   :requires-grad t)
	  (bias-of self)
	  (when bias
	    ;; Biases are (out-channels) tensor which sampled from U(-sqrt(k), sqrt(k))
	    (uniform-random `(,out-channels) (- (sqrt k)) (sqrt k) :requires-grad t)))))

(defmethod apply-conv2d ((self Conv2D) input)
  (with-slots ((stride stride) (padding padding) (dilation dilation) (weight weight) (bias bias) (groups groups) (kernel-size kernel-size)) self
    (multiple-value-bind (N C-In H-in W-in) (apply #'values (shape input))
      (declare (ignore C-in))
      (multiple-value-bind (C-Out Icg K-h K-w) (apply #'values (shape weight))
	(declare (ignore Icg K-h K-w))
	(let ((H-out (conv-out-size H-in (car padding) (car dilation) (car kernel-size) (car stride)))
	      (W-out (conv-out-size W-in (second padding) (second dilation) (second kernel-size) (second stride))))
	  (let* ((col   (unfold input dilation kernel-size stride padding))
		 (col-w (!reshape weight c-out t))
		 (out   (!matmul col (!t col-w)))
		 (out   (if bias
			    (!add out (%transform bias[i] -> [~ i]))
			    out)))
	    (!permute (!reshape out N h-out w-out C-out) 3 0 2 1)))))))

