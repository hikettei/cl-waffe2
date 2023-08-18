
(in-package :cl-waffe2)

;; Here's a utility macro which configurates vm/node's setting.

;; TODO: Place here, with-devices
(defmacro with-cpu (&body body)
  "TODO: Docstring"
  #+sbcl
  `(with-devices (CPUTensor LispTensor)
     ,@body)
  #-sbcl
  `(with-devices (LispTensor)
     ,@body))

;; (defmacro with-cuda)

(defmacro with-dtype (dtype &body body)
  `(let ((*default-dtype* ,dtype))
     ,@body))

(defmacro with-column-major (&body body)
  `(let ((*default-order* :column))
     ,@body))

(defmacro with-row-major (&body body)
  `(let ((*default-order* :row))
     ,@body))

;; Broadcast_Auto shouldn't be modular, all the nodes defined in cl-waffe2, should work under all combines of config.
(defmacro with-config ((&key
			  ;; TO ADD:
			  ;; Global Dtype
			  ;; (sin uint8) -> Global Float Dtype
			  ;; Matmul-Accuracy
			  ;; 
			  (device :cpu)
			  (no-grad nil)
			  (dtype :float)
			  (order :column)
			  (num-cores 1))
		       &body
			 body)
  "Integrates all the annoying configs."
  (declare (type (and keyword (member :cpu :cuda)) device)
	   (type (and keyword (member :row :column)) order)
	   (type keyword dtype))
  `(,(case device
       (:cpu 'with-cpu)
       (:cuda 'with-cuda))
    (,(if no-grad
	  'with-no-grad
	  'progn)
     (,(case order
	 (:column 'with-column-major)
	 (:row    'with-row-major))
      (with-dtype ,dtype
	(with-num-cores (,num-cores)
	  ,@body))))))


;; TODO: Add set-config for REPL.



(defun collect-initarg-slots (slots constructor-arguments)
  (map 'list #'(lambda (slots)
		 ;; Auto-Generated Constructor is Enabled Only When:
		 ;; slot has :initarg
		 ;; slot-name corresponds with any of constructor-arguments
		 (when
		     (and
		      (find (first slots) (alexandria:flatten constructor-arguments))
		      (find :initarg slots))
		   slots))
       slots))


(defun set-devices-toplevel (&rest devices)
  "
## [function] set-devices-toplevel

```lisp
(set-devices-toplevel &rest devices)
```

Declares devices to use.
"
  (assert (every #'(lambda (x) (subtypep x 'AbstractTensor)) devices)
	  nil
	  "set-devices-toplevel: the given device isn't subtype of AbstractTensor: ~a" devices)
  
  (setf cl-waffe2/vm.generic-tensor:*using-backend* devices))

(defun find-available-backends (&optional (from (find-class 'cl-waffe2/vm.generic-tensor:AbstractTensor)))
  (let ((classes (c2mop:class-direct-subclasses from)))
    (map 'list
	 #'(lambda (class-from)
	     `(,class-from ,@(find-available-backends class-from)))
	 classes)))

(defun rendering-backends-tree-to (tree-top out)
  (labels ((rendering-helper (tree-from indent-level)
	     (format out "~%")
	     (dotimes (i indent-level) (princ " " out))
	     (let ((using-p (find (format nil "~a" (class-name (car tree-from)))
				  *using-backend*
				  :key #'symbol-name
				  :test #'equal)))
	       (format out "~a~a~a: ~a"
		       ;; Now the computation is done under the device?
		       ;; -> If so, add *
		       (if (= indent-level 0)
			   ""
			   "└")
		       (if using-p
			   "[*]"
			   "[-]")
		       (class-name (car tree-from))
		       (current-backend-state (class-name (car tree-from))))
	       (let ((indent-level (+ indent-level 4)))
		 (dolist (more (cdr tree-from))
		   (rendering-helper more indent-level))))))
    (rendering-helper tree-top 0)))

(defun show-backends (&key (stream t))
  "
## [function] show-backends

```lisp
(show-backends &key (stream t))
```

collects and displays the current state of devices to the given `stream`

### Example

```lisp
(show-backends)

─────[All Backends Tree]──────────────────────────────────────────────────

[*]CPUTENSOR: OpenBLAS=available *simd-extension-p*=available
    └[-]JITCPUTENSOR: compiler=gcc flags=(-fPIC -O3 -march=native) viz=NIL

[*]LISPTENSOR: Common Lisp implementation on matrix operations
    └[-]JITLISPTENSOR: To be deleted in the future release. do not use this.

[-]SCALARTENSOR: is a special tensor for representing scalar values.
    └[-]JITCPUSCALARTENSOR: Use with JITCPUTensor

([*] : in use, [-] : not in use.)
Add a current-backend-state method to display the status.
─────[*using-backend*]───────────────────────────────────────────────────

Priority: Higher <───────────────────>Lower
                  CPUTENSOR LISPTENSOR 

(use with-devices macro or set-devices-toplevel function to change this parameter.)
```
"

  (format stream "~%~a"
	  (with-output-to-string (out)
	    (let ((backends-tree (find-available-backends)))
	      (dotimes (i 5)  (princ "─" out))
	      (princ "[All Backends Tree]" out)
	      (dotimes (i 50) (princ "─" out))
	      (mapc #'(lambda (tree)
			(format out "~%")
			(rendering-backends-tree-to tree out))
		    backends-tree)
	      (format out "~%~%([*] : in use, [-] : not in use.)")
	      (format out "~%Add a current-backend-state method to display the status.~%")

	      
	      (dotimes (i 5)  (princ "─" out))
	      (princ "[*using-backend*]" out)
	      (dotimes (i 51) (princ "─" out))

	      (format out "~%~%")
	      (let ((total-namelen 0))
		(dolist (name *using-backend*)
		  (incf total-namelen (length (symbol-name name))))

		(format out "Priority: Higher <")
		(dotimes (i total-namelen) (princ "─" out))
		(format out ">Lower~%")
		(format out "                  ")
		(dolist (name *using-backend*)
		  (princ (symbol-name name) out)
		  (princ " " out))

		(format out "~%~%(use with-devices macro or set-devices-toplevel function to change this parameter.)~%"))))))