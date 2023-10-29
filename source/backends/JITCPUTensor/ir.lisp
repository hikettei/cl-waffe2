
(in-package :cl-waffe2/backends.jit.cpu)

(defun render-instruction (instruction indices)
  (declare (type Instruction instruction)
	   (type list indices))

  (with-slots ((type type) (displace-to displace-to) (args args) (fname fname)) instruction
    (case type
      (:modify
       ;; modify: A fname B
       (write-c-line "~a ~a ~a;~%"
		     (cAref displace-to indices)
		     fname		     
		     (apply
		      #'concatenate
		      'string
		      (butlast
		       (loop for arg in args
			     append
			     (list (cAref arg indices) ", "))))))
      (:apply
       ;; apply: A = fname(B)
       (write-c-line "~a ~a ~a;~%"
		     (cAref displace-to indices)
		     fname
		     (apply
		      #'concatenate
		      'string
		      (butlast
		       (loop for arg in args
			     append
			     (list (cAref arg indices) ", "))))))
      (T
       (error "Unknown instruction type: ~a" instruction)))))

