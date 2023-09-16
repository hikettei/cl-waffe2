
(in-package :cl-waffe2/viz)

;; Print out to terminal

(defparameter *indent-level* 0)
(defparameter *indent-with* " ")

(defun dprint (toplevel &key (stream t) (fuse-p t) (backward t) (print-device t) (indent-width 2) (initial-indent 0) (max-count nil) &aux (seen nil) (count 0))
  "
## [function] dprint

```lisp
(dprint toplevel &key (stream t) (fuse-p t) (backward t) (print-device t) (indent-width 2) (initial-indent 0) (max-count nil))
```
"
  (declare (type AbstractNode))
  (let ((*indent-level* initial-indent))
    
    (labels ((indent (stream)
	       (if (= *indent-level* 0)
		   nil
		   (progn
		     (dotimes (i (1- *indent-level*)) (princ *indent-with* stream))
		     (princ "|" stream))))
	     (print-tensor (stream tensor)
	       (let* ((tensor-name (format nil "<~a:~a>"
					   (if (slot-value tensor 'requires-grad)
					       "Param"
					       (if (eql (tensor-attribute tensor) :input)
						   "TMP"
						   "Input"))
					   (class-name (class-of tensor))))
		      (tensor-info (format nil "~a~a"
					   (tensor-id tensor)
					   (if (scalar-p tensor)
					       "(1)"
					       (shape tensor)))))
		 (indent stream)
		 (format stream "~a~a~%" tensor-name tensor-info)))
	     
	     (print-edge (stream tensor)
	       (let* ((node-name (cl-ppcre:split "-" (format nil "~a" (class-name (class-of (tensor-backward tensor))))))
		      (op-name (apply
				#'concatenate
				'string
				(butlast node-name)))
		      (device-name (if print-device (format nil "~a" (class-name (class-of tensor))) ""))
		      (line1       (format nil "Op:~a{~a}" op-name device-name)))
		 (indent stream)
		 (format stream "~a~%" line1)))

	     (explore-node (stream tensor)
	       (when (null (find (tensor-iid tensor) seen))
		 (incf count)
		 (when (and max-count
			    (>= count max-count))
		   (return-from explore-node))
		 (push (tensor-iid tensor) seen)
		 (if (or (detach-p tensor)
			 (null (tensor-backward tensor)))
		     (print-tensor stream tensor)
		     (print-edge stream tensor)))

	       (let ((*indent-level* (+ indent-width *indent-level*)))
		 (dolist (var (tensor-variables tensor))
		   (explore-node stream var)))))
      

      (format stream "~%source:~%~a"
	      (with-output-to-string (out)
		(explore-node out toplevel)))

      ;; Disassembled waffe2 iseq can't be displayed because it is too much complicated and nested
      (cl-waffe2/vm:disassemble-waffe2-ir toplevel :backward backward :stream stream :fuse-p fuse-p) 
      toplevel)))
