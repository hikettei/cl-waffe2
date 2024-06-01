
(in-package :cl-waffe2/backends.aten)

(defstruct (Blueprint
	    (:conc-name bp-))
  (code "" :type string)
  (deps nil :type list))

(defun gid (rank)
  "generates a symbol: _GID_rank"
  (intern (format nil "_GID_~a" (code-char (+ 65 rank)))))

(defun render-shape (tensor &key (f #'shape))
  (apply
   #'concatenate
   'string
   (butlast
    (loop for s in (funcall f tensor)
	  append (list (format nil "~a" s) " ")))))

(defun tensor->shape-tracker (tensor)
  (if (scalar-p tensor)
      (coerce (format nil "~a{~a}[]<>()" (tensor-id tensor) (dtype tensor)) '(simple-array character (*)))
      (coerce (format nil "~a{~a}[~a]<~a>()" (tensor-id tensor) (dtype tensor) (render-shape tensor) (render-shape tensor :f #'cl-waffe2/vm.generic-tensor::tensor-permute-order)) '(simple-array character (*)))))

(defun lazy-stride (tensor axis)
  (declare (type fixnum axis))
  (labels ((expand-helper (stride)
	     (trivia:ematch stride
	       ((type fixnum) stride)
	       ((type symbol) (intern (format nil "~a" stride)))
	       ((list 'the 'fixnum form)
		(expand-helper form))
	       (_
		(if (eql (car stride) 'cl-waffe2/vm::maybe-observe-axis)
		    (expand-helper (second (second stride)))
		    (map 'list #'expand-helper stride))))))
    (expand-helper (nth axis (tensor-stride tensor)))))

(defun lazy-stride-depends-on (tensor axis)
  (declare (type fixnum axis))
  (labels ((expand-helper (stride)
	     (trivia:match stride
	       ((type symbol) (list stride))
	       ((list 'the 'fixnum form)
		(expand-helper form))
	       ((type list)
		(if (eql (car stride) 'cl-waffe2/vm::maybe-observe-axis)
		    (expand-helper (second (second stride)))
		    (map 'list #'expand-helper stride))))))
    (loop for x in (expand-helper (nth axis (tensor-stride tensor)))
	  if x collect x)))

(defun lazy-aref-form (tensor indices batch-p)
  (if (scalar-p tensor)
      (intern (symbol-name (tensor-id tensor)))
      `(aref ,(intern (symbol-name (tensor-id tensor)))
	     (+
	      ,@(loop for dim upfrom 0 below (dims tensor)
		      for view   in (tensor-view tensor)
		      for index in indices
		      for stride = (lazy-stride tensor dim)
		      collect
		      (let ((range (subscript-range view)))
			(if (subscript-broadcast view)
			    0
			    `(*
			      ,@(unless batch-p (list stride))
			      ,(wf/iter:range-step range)
			      ,index))))))))

(defun collect-depends-on (tensors)
  (flet ((depends-on (tensor)
	   (let ((list `(,@(loop for i upfrom 0 below (dims tensor)
				 append (lazy-stride-depends-on tensor i))
			 ,@(shape tensor))))
	     (loop for l in list
		   if (symbolp l)
		     collect l))))
    (remove-duplicates
     (apply
      #'append
      (map
       'list
       #'depends-on
       tensors)))))

(defun unary->aten-code (tensors rank op)
  (let* ((compiled-loops (solve-loop-order tensors rank nil))
	 (loop-depends)
	 (ids)
	 (collapsed-p (not (= (length compiled-loops) (dims (car tensors))))))
    (labels ((expand (nth)
	       (let ((subject (nth nth compiled-loops)))
		 (when subject
		   (if (eql (aloop-mode subject) :batch)
		       (progn
			 (push (gid nth) ids)
			 (when (not (numberp (aloop-size subject)))
			   (push (aloop-size subject) loop-depends))
			 `(loop (,(gid nth) 0 ,(aloop-size subject) ,(aloop-by subject))
				,(expand (1+ nth))))
		       (progn
			 (push (gid nth) ids)
			 (let ((args (map 'list #'(lambda (x) (lazy-aref-form x (reverse ids) (and (eql (aloop-mode subject) :apply-flatten) collapsed-p))) tensors)))
			   (if (eql (aloop-mode subject) :apply)
			       (when (not (numberp (aloop-element-n subject)))
				 (push (aloop-element-n subject) loop-depends))
			       (when (not (numberp (aloop-size subject)))
				 (push (aloop-size subject) loop-depends)))
			   `(loop (,(gid nth)
				   0
				   ,(if (eql (aloop-mode subject) :apply)
					(aloop-element-n subject)
					(aloop-size subject))
				   ,(aloop-by subject))
				  ,@(apply op args)))))))))
      (let ((code (format nil "~a" (expand 0)))
	    (depends-on (remove-duplicates `(,@(collect-depends-on tensors) ,@loop-depends))))
	(make-blueprint
	 :code code
	 :deps depends-on)))))

