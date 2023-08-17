
(in-package :cl-waffe2-simd)

(defmacro uformat (dest arg &rest args)
  `(string-upcase (format ,dest ,arg ,@args)))

(macrolet ((define-arith-op (opname)
	     `(progn
		;;size xptr incx xptr incy
		,@(loop for prefix in `(:d :s :i32 :i16 :i8 :u32 :u16 :u8)
			for dtype  in `(:double :float :int32 :int16 :int8 :uint32 :uint16 :uint8)
			collect
			`(progn
			   (export ',(intern (uformat nil "waffe2-~a~a" prefix opname)))
			   (declaim (inline ,(intern (uformat nil "waffe2-~a~a" prefix opname))))
			   (defcfun ,(uformat nil "waffe2_~a~a" prefix opname) :void
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

(macrolet ((define-inv-family ()
	     `(progn
		;;size xptr incx xptr incy
		,@(loop for prefix in `(:d :s :i32 :i16 :i8 :u32 :u16 :u8)
			for dtype  in `(:double :float :int32 :int16 :int8 :uint32 :uint16 :uint8)
			for ltype  in `(double-float single-float (signed-byte 32) (signed-byte 16) (signed-byte 8) (unsigned-byte 32) (unsigned-byte 16) (unsigned-byte 8))
			collect
			`(progn
			   (export ',(intern (uformat nil "waffe2-~ainv" prefix)))
			   (declaim (inline ,(intern (uformat nil "waffe2-~ainv" prefix))))
			   (defun ,(intern (uformat nil "waffe2-~ainv" prefix))
			       (n x incx)
			     (let ((y (make-array 1 :initial-element (coerce 1 ',ltype) :element-type ',ltype)))
			       (with-pointer-to-vector-data (y y)
				 (,(intern (uformat nil "waffe2-~adiv" prefix))
				  n
				  y
				  0
				  x
				  incx)))))))))
  (define-inv-family))

(macrolet ((define-maxmin-op (opname)
	     `(progn
		;;size xptr incx xptr incy
		,@(loop for prefix in `(:d :s :i32 :i16 :i8 :u32 :u16 :u8)
			for dtype  in `(:double :float :int32 :int16 :int8 :uint32 :uint16 :uint8)
			collect
			`(progn
			   (export ',(intern (uformat nil "waffe2-~a~a" prefix opname)))
			   (declaim (inline ,(intern (uformat nil "waffe2-~a~a" prefix opname))))
			   (defcfun ,(uformat nil "waffe2_~a~a" prefix opname) :void
			     (n :long)
			     (x (:pointer ,dtype))
			     (incx :long)
			     (y (:pointer ,dtype))
			     (incy :long)))))))
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
			   (defcfun ,(uformat nil "waffe2_~a~a" prefix opname) :void
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
			   (defcfun ,(uformat nil "waffe2_~a~a_scal" prefix opname) :void
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

;; ScalarAdd

;; Math APIs sin/cos/tan/exp/log/abs/sign etc...

;; Unfold
