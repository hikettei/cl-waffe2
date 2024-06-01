
(in-package :cl-waffe2/backends.aten)

(defun gid (rank)
  "generates a symbol: _GID_rank"
  (intern (format nil "_GID_~a" (code-char (+ 65 rank)))))

(defun tensor->shape-tracker (tensor &key (flatten-mode t))
  (if flatten-mode
      (if (scalar-p tensor)
	  (coerce (format nil "~a{~a}[]<>()" (tensor-id tensor) (dtype tensor)) '(simple-array character (*)))
	  (coerce (format nil "~a{~a}[~a]<0>()" (tensor-id tensor) (dtype tensor) (apply #'* (shape tensor))) '(simple-array character (*))))
      (error "not ready")))

(defun lazy-aref-form (tensor indices batch-p)
  (if (scalar-p tensor)
      (intern (symbol-name (tensor-id tensor)))
      `(aref ,(intern (symbol-name (tensor-id tensor)))
	     (+
	      ,@(loop for dim upfrom 0 below (dims tensor)
		      for stride in (tensor-stride tensor)
		      for view   in (tensor-view tensor)
		      for index in indices
		      collect
		      (let ((range (subscript-range view)))
			(if (subscript-broadcast view)
			    0
			    `(*
			      ,@(unless batch-p (list stride))
			      ,(wf/iter:range-step range)
			      ,index))))))))

(defun unary->aten-code (tensors rank op)
  (let* ((compiled-loops (solve-loop-order tensors rank nil))
	 (ids)
	 (collapsed-p (not (= (length compiled-loops) (dims (car tensors))))))
    (labels ((expand (nth)
	       (let ((subject (nth nth compiled-loops)))
		 (when subject
		   (if (eql (aloop-mode subject) :batch)
		       (progn
			 (push (gid nth) ids)
			 `(loop (,(gid nth) 0 ,(aloop-size subject) ,(aloop-by subject))
				,(expand (1+ nth))))
		       (progn
			 (push (gid nth) ids)
			 (let ((args (map 'list #'(lambda (x) (lazy-aref-form x (reverse ids) collapsed-p)) tensors)))
			   `(loop (,(gid nth) 0 ,(aloop-size subject) ,(aloop-by subject))
				  ,@(apply op args)))))))))
      (format nil "~a" (expand 0)))))

