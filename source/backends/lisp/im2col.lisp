
(in-package :cl-waffe2/backends.lisp)


(define-with-typevar (im2col-caller u)
    (data-col N C H W output-h output-w K-H K-W PAD-H PAD-W stride-h stride-w dilation-h dilation-w data-im)
  (declare (optimize (speed 3))
           (type AbstractTensor data-col data-im)
	   (type (unsigned-byte 32) N C H W output-h output-w k-H K-W PAD-H PAD-W stride-h stride-w dilation-h dilation-w))

  (with-facets ((data-col* (data-col :direction 'simple-array :sync t))
		(data-im*  (data-im :direction  'simple-array :sync t)))
    (declare (type (simple-array u (*)) data-col* data-im*))
    
    (macrolet ((%* (a b)
		 `(the (unsigned-byte 32) (* (the (unsigned-byte 32) ,a) (the (unsigned-byte 32) ,b))))
	       (%+ (&rest numbers)
		 `(the (unsigned-byte 32) (+ ,@numbers))))
      
      (let* ((zero         (coerce 0 (quote u)))
	     (channel-cols (%* k-w (%* C k-h)))
	     (offset-im  0)
	     (offset-col 0)
	     (im-stride  (car (tensor-stride data-im)))
	     (col-stride (car (tensor-stride data-col))))
	(declare (type (unsigned-byte 32) im-stride col-stride offset-im offset-col channel-cols))
	  
	(dotimes (i (array-total-size data-im*))
	  (setf (aref data-im* i) zero))

	(dotimes (batch N)
	  (dotimes (c-col channel-cols)
	    (let ((w-offset (mod c-col K-W))
		  (h-offset (mod (/ c-col K-W) K-H))
		  (c-im     (/
			     (the (unsigned-byte 32)
				  (/
				   (the (unsigned-byte 32) c-col) K-H))
			     K-W)))
	      (declare (type (unsigned-byte 32) w-offset h-offset c-im))

	      (dotimes (h-col output-h)
		(let ((h-im (%+ (%* h-col stride-h)
				(%* h-offset dilation-h)
				(- pad-h))))
		  (declare (type (unsigned-byte 32) h-im))
		  (when (and (>= h-im 0)
			     (<  h-im H))
		    (dotimes (w-col output-w)
		      (let ((w-im (%+ (%* w-col stride-w)
				      (%* w-offset dilation-w)
				      (- pad-w))))
			(declare (type (unsigned-byte 32) w-im))
			(when (and (>= w-im 0)
				   (<  w-im W))
			  (setf
			   (aref data-col* (%+ w-col (%* output-w (%+ h-col (%* c-col output-h)))))
			   (aref data-im*  (%+ offset-im w-im (%* W (%+ h-im (%* c-im H))))))))))))))
	  (incf offset-im im-stride)
	  (incf offset-col col-stride))))
    nil))


(define-with-typevar (col2im-caller u)
    (data-col N C H W output-h output-w K-H K-W PAD-H PAD-W stride-h stride-w dilation-h dilation-w data-im)
  (declare (optimize (speed 3))
           (type AbstractTensor data-col data-im)
	   (type (unsigned-byte 32) N C H W output-h output-w k-H K-W PAD-H PAD-W stride-h stride-w dilation-h dilation-w))

  (with-facets ((data-col* (data-col :direction 'simple-array :sync t))
		(data-im*  (data-im :direction  'simple-array :sync t)))
    (declare (type (simple-array u (*)) data-col* data-im*))
    
    (macrolet ((%* (a b)
		 `(the (unsigned-byte 32) (* (the (unsigned-byte 32) ,a) (the (unsigned-byte 32) ,b))))
	       (%+ (&rest numbers)
		 `(the (unsigned-byte 32) (+ ,@numbers))))
      
      (let* ((zero         (coerce 0 (quote u)))
	     (channel-cols (%* k-w (%* C k-h)))
	     (offset-im  0)
	     (offset-col 0)
	     (im-stride  (car (tensor-stride data-im)))
	     (col-stride (car (tensor-stride data-col))))
	(declare (type (unsigned-byte 32) im-stride col-stride offset-im offset-col channel-cols))
	
	(dotimes (i (array-total-size data-im*))
	  (setf (aref data-im* i) zero))

	(dotimes (batch N)
	  (dotimes (c-col channel-cols)
	    (let ((w-offset (mod c-col K-W))
		  (h-offset (mod (/ c-col K-W) K-H))
		  (c-im     (/
			     (the (unsigned-byte 32)
				  (/
				   (the (unsigned-byte 32) c-col) K-H))
			     K-W)))
	      (declare (type (unsigned-byte 32) w-offset h-offset c-im))

	      (dotimes (h-col output-h)
		(let ((h-im (%+ (%* h-col stride-h)
				(%* h-offset dilation-h)
				(- pad-h))))
		  (declare (type (unsigned-byte 32) h-im))
		  (when (and (>= h-im 0)
			     (<  h-im H))
		    (dotimes (w-col output-w)
		      (let ((w-im (%+ (%* w-col stride-w)
				      (%* w-offset dilation-w)
				      (- pad-w))))
			(declare (type (unsigned-byte 32) w-im))
			(when (and (>= w-im 0)
				   (<  w-im W))
			  (incf
			   (aref data-im*  (%+ offset-im  w-im (%* W (%+ h-im (%* c-im H)))))
			   (aref data-col* (%+ offset-col w-col (%* output-w (%+ h-col (%* c-col output-h))))))))))))))
	  (incf offset-im im-stride)
	  (incf offset-col col-stride))))
    nil))


(define-impl-op (Im2ColNode :device LispTensor)
		:forward ((self x col)
			  (setf (h-of self) (nth 2 (shape x))
				(w-of self) (nth 3 (shape x)))
			  (with-slots ((N N) (C C) (H H) (W W)				       
				       (h-out h-out) (w-out w-out)
				       (k-h k-h) (k-w k-w)
				       (padding-h padding-h) (padding-w padding-w)
				       (dilation-h dilation-h) (dilation-w dilation-w)
				       (stride-h stride-h) (stride-w stride-w))
			      self
			    ;; (data-col N C H W output-h output-w K-H K-W PAD-H PAD-W stride-h stride-w dilation-h dilation-w data-im)
			    (funcall (im2col-caller (dtype x))
				     col
				     N C H W
				     h-out w-out
				     k-h k-w
				     padding-h padding-w
				     stride-h stride-w
				     dilation-h dilation-w
				     x)
			    x)))

(define-impl-op (Col2ImNode :device LispTensor)
		:forward ((self x col)
			  (setf (h-of self) (nth 2 (shape x))
				(w-of self) (nth 3 (shape x)))
			  (with-slots ((N N) (C C) (H H) (W W)				       
				       (h-out h-out) (w-out w-out)
				       (k-h k-h) (k-w k-w)
				       (padding-h padding-h) (padding-w padding-w)
				       (dilation-h dilation-h) (dilation-w dilation-w)
				       (stride-h stride-h) (stride-w stride-w))
			      self
			    ;; (data-col N C H W output-h output-w K-H K-W PAD-H PAD-W stride-h stride-w dilation-h dilation-w data-im)
			    (funcall (col2im-caller (dtype x))
				     col
				     N C H W
				     h-out w-out
				     k-h k-w
				     padding-h padding-w
				     stride-h stride-w
				     dilation-h dilation-w
				     x)
			    (values col))))

