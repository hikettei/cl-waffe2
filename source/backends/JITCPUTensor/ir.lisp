
(in-package :cl-waffe2/backends.jit.cpu)

(defun render-instruction (instruction indices &optional (place-holders nil))
  (declare (type Instruction instruction)
	   (type list indices))

  (with-slots ((type type) (displace-to displace-to) (args args) (fname fname)) instruction
    (flet ((place-holder-p (tensor)
	     (and place-holders
		  (let ((result (gethash (tensor-id tensor) place-holders)))
		    (and
		     result
		     (if (functionp result)
			 (funcall result)
			 result))))))
      (case type
	(:modify
	 ;; modify: A fname B
	 (write-c-line "~a ~a ~a;~%"
		       (or (place-holder-p displace-to) (cAref displace-to indices))
		       fname		     
		       (apply
			#'concatenate
			'string
			(butlast
			 (loop for arg in args
			       append
			       (list (or (place-holder-p arg) (cAref arg indices)) ", "))))))
	(:apply
	 ;; apply: A = fname(B)
	 (write-c-line "~a ~a ~a;~%"
		       (or (place-holder-p displace-to) (cAref displace-to indices))
		       fname
		       (apply
			#'concatenate
			'string
			(butlast
			 (loop for arg in args
			       append
			       (list (or (place-holder-p arg) (cAref arg indices)) ", "))))))
	(T
	 (error "Unknown instruction type: ~a" instruction))))))

