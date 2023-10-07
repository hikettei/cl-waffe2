
(in-package :cl-waffe2/backends.lisp)

;; Mem: https://ieeexplore.ieee.org/document/9342343

;; [TODO_LIST]
;; - 1. Paddingの修正(OK)
;; - 2. LispTensorのUnfoldを動かす im2col -> OK col2im -> ?
;; - 3. CPUTensorのUnfold(mergeの後で)
;; - 4. GhActions ... SIMD Extension Build test...
;; - 5. 
;; - 6. Dynamic Shaping??

(define-with-typevar (im2col-caller u) (data-col strides1 N C out-h out-w K-h K-w Pad-H Pad-W Stride-H Stride-W dilation-H dilation-W data-im strides2)
  (declare (optimize (speed 3))
	   (type (simple-array u (*)) data-im data-col)
	   (type (unsigned-byte 32) N C out-h out-w K-h K-w Pad-H Pad-W Stride-H Stride-W dilation-H dilation-w))
  (macrolet ((index (index nth stride-from)
	       `(the (unsigned-byte 32)
		     (* (the (unsigned-byte 32) ,index)
			(the (unsigned-byte 32)
			     (nth ,nth ,stride-from))))))
    (dotimes (h-start k-h)
      (let ((h-end (+
		    (the (unsigned-byte 32) (* dilation-h h-start))
		    (* stride-h out-h)
		    (- pad-H))))
	(declare (type (unsigned-byte 32) h-end))
	(dotimes (w-start k-w)
	  (let ((w-end (+ (the (unsigned-byte 32) (* dilation-w w-start))
			  (* stride-w out-w)
			  (- pad-w))))
	    (declare (type (unsigned-byte 32) w-end))
	    (loop for h-out-col fixnum upfrom 0
		  for h-out-im  fixnum upfrom h-start below h-end by stride-H do
		    (loop for w-out-col fixnum upfrom 0
			  for w-out-im  fixnum upfrom w-start below w-end by stride-W
			  do (dotimes (ni N)
			       (dotimes (ci C)
				 (setf (aref data-col
					     (+
					      (index ni 0 strides1)
					      (index ci 1 strides1)
					      (index h-start 2 strides1)
					      (index w-start 3 strides1)
					      (index h-out-col 4 strides1)
					      (index w-out-col 5 strides1)))
				       (aref data-im
					     (+
					      (index ni 0       strides2)
					      (index ci 1       strides2)
					      (index h-out-im 2 strides2)
					      (index w-out-im 3 strides2))))))))))))))

(define-with-typevar (col2im-caller u) (data-col strides1 N C out-h out-w K-h K-w Pad-H Pad-W Stride-H Stride-W dilation-H dilation-W data-im strides2)
  (declare (optimize (speed 3))
	   (type (simple-array u (*)) data-im data-col)
	   (type (unsigned-byte 32) N C out-h out-w K-h K-w Pad-H Pad-W Stride-H Stride-W dilation-H dilation-w))

  (macrolet ((index (index nth stride-from)
	       `(the (unsigned-byte 32)
		     (* (the (unsigned-byte 32) ,index)
			(the (unsigned-byte 32)
			     (nth ,nth ,stride-from))))))
    
    (let ((zero (coerce 0 (quote u))))
      (dotimes (i (array-total-size data-im)) (setf (aref data-im i) zero)))
    
    (dotimes (h-start k-h)
      (let ((h-end (+
		    (the (unsigned-byte 32) (* dilation-h h-start))
		    (* stride-h out-h)
		    (- pad-H))))
	(declare (type (unsigned-byte 32) h-end))
	(dotimes (w-start k-w)
	  (let ((w-end (+ (the (unsigned-byte 32) (* dilation-w w-start))
			  (* stride-w out-w)
			  (- pad-w))))
	    (declare (type (unsigned-byte 32) w-end))
	    (loop for h-out-col fixnum upfrom 0
		  for h-out-im  fixnum upfrom h-start below h-end by stride-H do
		    (loop for w-out-col fixnum upfrom 0
			  for w-out-im  fixnum upfrom w-start below w-end by stride-W
			  do (dotimes (ni N)
			       (dotimes (ci C)
				 (incf  (aref data-im
					      (+
					       (index ni 0       strides2)
					       (index ci 1       strides2)
					       (index h-out-im 2 strides2)
					       (index w-out-im 3 strides2)))
					(aref data-col
					      (+
					       (index ni 0 strides1)
					       (index ci 1 strides1)
					       (index h-start 2 strides1)
					       (index w-start 3 strides1)
					       (index h-out-col 4 strides1)
					       (index w-out-col 5 strides1))))))))))))))

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
			    (cl-waffe2:with-facets ((x*   (x :direction 'simple-array :sync nil))
						    (col* (col :direction 'simple-array :sync nil)))
			      (funcall (im2col-caller (dtype x))
				       col* (tensor-stride col)
				       N C
				       h-out w-out
				       k-h k-w
				       padding-h padding-w
				       stride-h stride-w
				       dilation-h dilation-w
				       x* (tensor-stride x)))
			    col)))

(define-impl-op (Col2ImNode :device LispTensor)
		:forward ((self col output-to)
			  (setf (h-of self) (nth 2 (shape col))
				(w-of self) (nth 3 (shape col)))
			  (with-slots ((N N) (C C) (H H) (W W)				       
				       (h-out h-out) (w-out w-out)
				       (k-h k-h) (k-w k-w)
				       (padding-h padding-h) (padding-w padding-w)
				       (dilation-h dilation-h) (dilation-w dilation-w)
				       (stride-h stride-h) (stride-w stride-w))
			      self
			    ;; (data-col N C H W output-h output-w K-H K-W PAD-H PAD-W stride-h stride-w dilation-h dilation-w data-im)
			    (cl-waffe2:with-facets ((o*   (output-to :direction 'simple-array :sync nil))
						    (col* (col :direction 'simple-array :sync nil)))
			      (funcall (im2col-caller (dtype col))
				       col* (tensor-stride col)
				       N C
				       h-out w-out
				       k-h k-w
				       padding-h padding-w
				       stride-h stride-w
				       dilation-h dilation-w
				       o* (tensor-stride output-to)))
			    output-to)))

