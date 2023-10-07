
(in-package :cl-waffe2-simd)

(defmacro uformat (dest arg &rest args)
  `(string-upcase (format ,dest ,arg ,@args)))

(defmacro dformat (dest arg &rest args)
  `(string-downcase (format ,dest ,arg ,@args)))

(defun make-fname (dtype name &key (scal nil))
  (intern (uformat nil "waffe2-~a~a~a"
		   (case dtype
		     (:double :d)
		     (:float  :s)
		     (:int32  :i32)
		     (:int16  :i16)
		     (:int8   :i8)
		     (:uint32 :u32)
		     (:uint16 :u16)
		     (:uint8  :u8))
		   name
		   (if scal
		       "-scal"
		       ""))
	  :cl-waffe2-simd))

(defun make-im2col (dtype order)
  (intern (uformat
	   nil
	   "waffe2-im2col-~a-~a"
	   (case dtype
	     (:double :d)
	     (:float  :s)
	     (:int32  :i)
	     (:int16  :i)
	     (:int8   :i)
	     (:uint32 :i)
	     (:uint16 :i)
	     (:uint8  :i))
	   (if (eql order :column)
	       "column"
	       "row"))
	  :cl-waffe2-simd))

(defun make-col2im (dtype order)
  (intern (uformat
	   nil
	   "waffe2-col2im-~a-~a"
	   (case dtype
	     (:double :d)
	     (:float  :s)
	     (:int32  :i)
	     (:int16  :i)
	     (:int8   :i)
	     (:uint32 :i)
	     (:uint16 :i)
	     (:uint8  :i))
	   (if (eql order :column)
	       "column"
	       "row"))
	  :cl-waffe2-simd))

(macrolet ((define-arith-op (opname)
	     `(progn
		;;size xptr incx xptr incy
		,@(loop for prefix in `(:d :s :i32 :i16 :i8 :u32 :u16 :u8)
			for dtype  in `(:double :float :int32 :int16 :int8 :uint32 :uint16 :uint8)
			collect
			`(progn
			   (export ',(intern (uformat nil "waffe2-~a~a" prefix opname)))
			   (declaim (inline ,(intern (uformat nil "waffe2-~a~a" prefix opname))))
			   (defcfun ,(dformat nil "waffe2_~a~a" prefix opname) :void
			     (n :long)
			     (x (:pointer ,dtype))
			     (incx :long)
			     (y (:pointer ,dtype))
			     (incy :long)))))))
  (define-arith-op add)
  (define-arith-op sub)
  (define-arith-op mul)
  (define-arith-op div)
  (define-arith-op copy))

;; Broadcasts scalar->tensor and applies op
(macrolet ((define-arith-bc-op (opname)
	     `(progn
		;;size xptr incx xptr incy
		,@(loop for prefix in `(:d :s :i32 :i16 :i8 :u32 :u16 :u8)
			for dtype  in `(:double :float :int32 :int16 :int8 :uint32 :uint16 :uint8)
			for ltype  in `(double-float single-float (signed-byte 32) (signed-byte 16) (signed-byte 8) (unsigned-byte 32) (unsigned-byte 16) (unsigned-byte 8))
			collect
			`(progn
			   (export ',(intern (uformat nil "waffe2-~a~a-scal" prefix opname)))
			   (declaim (inline ,(intern (uformat nil "waffe2-~a~a-scal" prefix opname))))
			   (defun ,(intern (uformat nil "waffe2-~a~a-scal" prefix opname))
			       (n x incx y-scalar)
			     (let ((y (make-array 1 :initial-element y-scalar :element-type ',ltype)))
			       (with-pointer-to-vector-data (y y)
				 (,(intern (uformat nil "waffe2-~a~a" prefix opname))
				  n
				  x
				  incx
				  y
				  0)))))))))
  (define-arith-bc-op add)
  (define-arith-bc-op sub)
  (define-arith-bc-op mul)
  (define-arith-bc-op div))


(macrolet ((define-inv (opname)
	     `(progn
		;;size xptr incx xptr incy
		,@(loop for prefix in `(:d :s :i32 :i16 :i8 :u32 :u16 :u8)
			for dtype  in `(:double :float :int32 :int16 :int8 :uint32 :uint16 :uint8)
			collect
			`(progn
			   (export ',(intern (uformat nil "waffe2-~a~a" prefix opname)))
			   (declaim (inline ,(intern (uformat nil "waffe2-~a~a" prefix opname))))
			   (defcfun ,(dformat nil "waffe2_~a~a" prefix opname) :void
			     (n :long)
			     (x (:pointer ,dtype))
			     (incx :long)))))))
  (define-inv inv))

(macrolet ((define-maxmin-op (opname)
	     `(progn
		;;size xptr incx xptr incy
		,@(loop for prefix in `(:d :s :i32 :i16 :i8 :u32 :u16 :u8)
			for dtype  in `(:double :float :int32 :int16 :int8 :uint32 :uint16 :uint8)
			collect
			`(progn
			   (export ',(intern (uformat nil "waffe2-~a~a" prefix opname)))
			   (declaim (inline ,(intern (uformat nil "waffe2-~a~a" prefix opname))))
			   (defcfun ,(dformat nil "waffe2_~a~a" prefix opname) :void
			     (n :long)
			     (x (:pointer ,dtype))
			     (incx :long)
			     (y (:pointer ,dtype))))))))
  (define-maxmin-op max)
  (define-maxmin-op min))

(macrolet ((define-cmp-op (opname)
	     `(progn
		;;size xptr incx xptr incy
		,@(loop for prefix in `(:d :s :i32 :i16 :i8 :u32 :u16 :u8)
			for dtype  in `(:double :float :int32 :int16 :int8 :uint32 :uint16 :uint8)
			collect
			`(progn
			   (export ',(intern (uformat nil "waffe2-~a~a" prefix opname)))
			   (declaim (inline ,(intern (uformat nil "waffe2-~a~a" prefix opname))))
			   (defcfun ,(dformat nil "waffe2_~a~a" prefix opname) :void
			     (n :long)
			     (x (:pointer ,dtype))
			     (incx :long)
			     (y (:pointer ,dtype))
			     (incy :long)
			     (out (:pointer ,dtype))
			     (inco :long)
			     (then ,dtype)
			     (else ,dtype)))))))
  ;; A=B
  (define-cmp-op eq)

  ;; A < B
  (define-cmp-op lt)

  ;; A <= B
  (define-cmp-op le)

  ;; A > B
  (define-cmp-op gt)

  ;; A >= B
  (define-cmp-op ge))

;; Scalar and matrix comparisons
(macrolet ((define-cmp-op (opname)
	     `(progn
		;;size xptr incx xptr incy
		,@(loop for prefix in `(:d :s :i32 :i16 :i8 :u32 :u16 :u8)
			for dtype  in `(:double :float :int32 :int16 :int8 :uint32 :uint16 :uint8)
			collect
			`(progn
			   (export ',(intern (uformat nil "waffe2-~a~a-scal" prefix opname)))
			   (declaim (inline ,(intern (uformat nil "waffe2-~a~a-scal" prefix opname))))
			   (defcfun ,(dformat nil "waffe2_~a~a_scal" prefix opname) :void
			     (n :long)
			     (x (:pointer ,dtype))
			     (incx :long)
			     (out (:pointer ,dtype))
			     (inco :long)
			     (y    ,dtype)
			     (then ,dtype)
			     (else ,dtype)))))))
  ;; A=scal
  (define-cmp-op eq)

  ;; A < scal
  (define-cmp-op lt)

  ;; A <= scal
  (define-cmp-op le)

  ;; A > scal
  (define-cmp-op gt)

  ;; A >= scal
  (define-cmp-op ge))

;; Math APIs sin/cos/tan/exp/log/abs/sign etc...
(macrolet ((define-math-op (opname)
	     `(progn
		;;size xptr incx xptr incy
		,@(loop for prefix in `(:d :s)
			for dtype  in `(:double :float)
			collect
			`(progn
			   (export ',(intern (uformat nil "waffe2-~a~a" prefix opname)))
			   (declaim (inline ,(intern (uformat nil "waffe2-~a~a" prefix opname))))
			   (defcfun ,(dformat nil "waffe2_~a~a" prefix opname) :void
			     (n :long)
			     (x (:pointer ,dtype))
			     (incx :long)
			     (y (:pointer ,dtype))
			     (incy :long)))))))
  (define-math-op sin)
  (define-math-op cos)
  (define-math-op tan)

  (define-math-op asin)
  (define-math-op acos)
  (define-math-op atan)

  (define-math-op sinh)
  (define-math-op cosh)
  (define-math-op tanh)

  (define-math-op asinh)
  (define-math-op acosh)
  (define-math-op atanh)

  (define-math-op log)
  (define-math-op log1p)
  (define-math-op log2)
  (define-math-op log10)

  (define-math-op abs)
  (define-math-op exp)
  (define-math-op sqrt)
  (define-math-op cbrt))

(export '(waffe2-sloge waffe2-dloge))
(defun waffe2-sloge (n x incx y incy) (waffe2-slog n x incx y incy))
(defun waffe2-dloge (n x incx y incy) (waffe2-dlog n x incx y incy))

(macrolet ((define-math-op1 (opname)
	     `(progn
		;;size xptr incx xptr incy
		,@(loop for prefix in `(:d :s)
			for dtype  in `(:double :float)
			collect
			`(progn
			   (export ',(intern (uformat nil "waffe2-~a~a" prefix opname)))
			   (declaim (inline ,(intern (uformat nil "waffe2-~a~a" prefix opname))))
			   (defcfun ,(dformat nil "waffe2_~a~a" prefix opname) :void
			     (n :long)
			     (x (:pointer ,dtype))
			     (incx :long)
			     (pow-n ,dtype)
			     (y (:pointer ,dtype))
			     (incy :long)))))))
  (define-math-op1 pow))

(macrolet ((define-unfolders (opname dtype-kw dtype order)
	     ;; waffe2_opname_dtype_order
	     (let ((name (uformat nil "waffe2-~a-~a-~a" opname dtype order)))
	       `(progn
		  (export ',(intern name))
		  (declaim (inline ,(intern name)))
		  (defcfun ,(dformat nil "waffe2_~a_~a_~a" opname dtype order) :void		   
		    (data-col (:pointer ,dtype-kw))
		    (N :long)
		    (C :long)
		    (H :long)
		    (W :long)
		    (H-out :long)
		    (W-out :long)
		    (K-h :long)
		    (K-w :long)
		    (pad-h :long)
		    (pad-w :long)
		    (stride-h :long)
		    (stride-w :long)
		    (dilation-h :long)
		    (dilation-w :long)
		    (data-im (:pointer ,dtype-kw)))))))
  (define-unfolders im2col :float s "column")
  (define-unfolders im2col :double d "column")
  (define-unfolders im2col :int i "column")
  (define-unfolders im2col :float s "row")
  (define-unfolders im2col :double d "row")
  (define-unfolders im2col :int i "row")

  (define-unfolders col2im :float s "column")
  (define-unfolders col2im :double d "column")
  (define-unfolders col2im :int i "column")
  (define-unfolders col2im :float s "row")
  (define-unfolders col2im :double d "row")
  (define-unfolders col2im :int i "row"))


