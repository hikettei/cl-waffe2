
(in-package :cl-waffe2/backends.lisp)

(defun iterator-symbols (rank)
  (loop for r upfrom 0 below rank
	collect
	(intern (format nil "~a~a" (gensym "L") (code-char (+ 65 r))))))

(defun lisp-name (value)
  (typecase value
    (symbol     
     `(the fixnum (cl-waffe2/vm:maybe-observe-axis ',value)))
    (T
     value)))

(defstruct (LazyLispInstruction
	    (:conc-name lli-)
	    (:constructor make-lli (opform out-to args &key (apply nil) (reduced-to 1))))
  "LazyLispInstruction, an blueprint for expand-iteration.
apply - Set to (apply function array). nil to (dotimes (...) ... )"
  (apply  apply  :type boolean)
  (out-to out-to :type AbstractTensor)
  (args   args   :type list)
  (opform opform :type (or function symbol))
  (reduced-to reduced-to))

(defun lli-form (reduced-to)
  (if (numberp reduced-to) reduced-to `(wf/vm:maybe-observe-axis ',reduced-to)))

(defun expand-iteration (tensors instructions index-components)
  (declare (type list tensors instructions))
  (let* ((abstract-loop (solve-loop-order tensors 1 T :mode :runtime))
	 (indices       (iterator-symbols (dims (car tensors)))))
    (labels ((expand-helper (rank)
	       (let ((aloop (nth rank abstract-loop))
		     (index (nth rank indices)))
		 (case (aloop-mode aloop)
		   (:batch
		    `(dotimes (,index ,(lisp-name (aloop-size aloop)))
		       ,(expand-helper (1+ rank))))
		   (T
		    `(progn
		       ,@(map
			  'list
			  #'(lambda (inst)
			      (expand-instruction
			       aloop
			       inst
			       index
			       indices
			       index-components))
			  instructions)))))))
      (expand-helper 0))))

(defun lazy-aref-form (tensor indices index-components)
  (let ((index `(the (unsigned-byte 32)
		     (+
		      ,@(loop for dim upfrom 0 below (dims tensor)
			      for stride in (tensor-actual-stride tensor)
			      for view   in (tensor-view tensor)
			      for index in indices
			      collect
			      (let ((range (subscript-range view)))
				(if (subscript-broadcast view)
				    0
				    `(the
				      (unsigned-byte 32)
				      ;; [FixME] Optimize the range-nth computation...
				      (*
				       ,stride
				       (wf/iter:range-nth ,range ,index))))))))))
    (if index-components index `(aref ,(tensor-id tensor) ,index))))

(defun expand-instruction (aloop instruction index-symbol indices index-components &aux (results (gensym)))
  (declare (type LazyLispInstruction instruction))
  (if (lli-apply instruction)
      `(let ((,results
	       (multiple-value-list
		(apply ,(lli-opform instruction)
		       ,@(loop for tensor in (lli-args instruction)
			       collect
			       `(loop for ,index-symbol of-type (unsigned-byte 32)
				      upfrom 0
					below ,(lisp-name (car (last (shape tensor))))
				      collect
				      ,(lazy-aref-form tensor indices index-components)))))))
	 (assert (= (length ,results) (the fixnum ,(lli-form (lli-reduced-to instruction))))
		 nil
		 "lazy-reduction: Assertion was failed. the function is expected to be returning ~a values but got ~a"
		 ,(lli-form (lli-reduced-to instruction))
		 (length ,results))
	 (dotimes (,index-symbol ,(lli-form (lli-reduced-to instruction)))
	   (setf ,(lazy-aref-form (lli-out-to instruction) indices nil) (nth ,index-symbol ,results))))
      `(dotimes (,index-symbol ,(lisp-name (aloop-element-n aloop)))
	 (setf ,(lazy-aref-form (lli-out-to instruction) indices nil)
	       (funcall
		,(lli-opform instruction)
		,@(loop for tensor in (lli-args instruction)
			collect
			(lazy-aref-form tensor indices index-components)))))))

(defun lazy-call-form (tensors instructions finally-return &key (index-components nil))
  `(progn
     (let* (,@(loop for tensor in tensors
		    collect
		    `(,(tensor-id tensor) (tensor-vec ,tensor))))
       (declare ,@(loop for tensor in tensors
			collect
			`(type (simple-array ,(dtype->lisp-type (dtype tensor)) (*)) ,(tensor-id tensor)))
		(optimize (speed 3)))
       ,(expand-iteration tensors instructions index-components))
     ,finally-return))

