
(in-package :cl-waffe2/vm.generic-tensor)


;; Printing Parameter for Preview
(defparameter *matrix-element-displaying-size* 13 ;; the same as digits of single-float
  "Decides how long elements to be omitted. If 3, cl-waffe prints 123 as it is but 1234 is displayed like: 1... (omitted)")

(defparameter *matrix-column-elements-displaying-length* 6
  "(1 2 3 4 5 6) -> (1 2 .. 5 6)")

(defparameter *matrix-columns-displaying-length* 4
  "((1 2 3) (4 5 6) (7 8 9)) -> ((1 2 3) ... (7 8 9))")

(defun trim-string (str max-length)
  "Trimming the given str within the length of max-length.
The result sequence MUST not over max-length.
... <- this part is not considered."
  (concatenate 'string
	       (subseq str 0 (min (length str) max-length))
	       (if (< max-length (length str))
		   "..."
		   "")))

(defun padding-str (str width &key (dont-fill nil))
  "Return a string whose length is equivalent to width."
  (let* ((trimmed-str (trim-string str (- width 3)))
	 (pad-size (- width (length trimmed-str))))
    (with-output-to-string (out)
      (format out "~a" trimmed-str)
      (unless dont-fill
	(loop repeat (- pad-size 3) do (format out " ")))
      out)))

(defun print-element (element &key (stream nil) (dont-fill nil))
  (format stream "~a"
	  (padding-str
	   (format nil "~a" element)
	   *matrix-element-displaying-size*
	   :dont-fill dont-fill)))

(defun last-mref (tensor index &aux (k (length (shape tensor))))
  (let ((sub (make-list k :initial-element 0)))
    (setf (nth (1- k) sub) index)
    (apply #'mref tensor sub)))

;; (1 2)
(defun pprint-1d-vector (stream
			 dim-indicator
			 tensor
			 &aux
			   (size (nth dim-indicator (shape tensor))))

  ;;"Assertion Failed with matrix.dims == 1")

  (if (>= size
	  *matrix-column-elements-displaying-length*)
      (let ((midpoint (round (/ *matrix-column-elements-displaying-length* 2))))
	(write-string "(" stream)
	(dotimes (k midpoint)
	  (write-string (print-element (last-mref tensor k)) stream)
	  (write-string " " stream))

	(write-string "~ " stream)

	(loop for k downfrom midpoint to 1
	      do (progn
		   (write-string
		    (print-element
		     (last-mref tensor (- size k))
		     :dont-fill (= k 1))
		    stream)
		   (unless (= k 1)
		     (write-string " " stream))))
	(write-string ")" stream))
      (progn
	(write-string "(" stream)
	(dotimes (i size)
	  (write-string (format nil "~A" (print-element (last-mref tensor i) :dont-fill (= i (1- size)))) stream)
	  (unless (= i (1- size))
	    (write-string " " stream)))
	(write-string ")" stream))))


;; More columns in one print
(defun pprint-vector (stream
		      tensor
		      &optional
			(newline T)
			(indent-size 0)
			(dim-indicator 0))
  (declare (type AbstractTensor tensor))
  (cond
    ((= 1 (length (shape tensor)))
     (pprint-1d-vector stream dim-indicator tensor))
    ((= dim-indicator (1- (length (shape tensor))))
     (pprint-1d-vector stream dim-indicator tensor))
    (T
     (write-string "(" stream)
     (if (< (nth dim-indicator (shape tensor))
	    *matrix-columns-displaying-length*)
	 ;; Can elements be printed at once?
	 (let ((first-dim (nth dim-indicator (shape tensor)))
	       (args      (make-list dim-indicator :initial-element t)))
	   (dotimes (i first-dim)
	     ;; pprint(n-1) and indent
	     (let ((tensor-view (apply #'view tensor `(,@args ,i))))
	       
	       (pprint-vector stream tensor-view newline (1+ indent-size) (1+ dim-indicator)))
	     
	     (unless (= i (1- first-dim))
	       (if newline
		   (progn
		     (write-char #\Newline stream)
		     ;; Rendering Indents
		     (dotimes (k (+ (1+ indent-size)))
		       (write-string " " stream)))
		   (write-string " " stream))))
	   (write-string ")" stream))
	 (let ((args (make-list dim-indicator :initial-element t))
	       (midpoint (round (/ *matrix-columns-displaying-length* 2))))
	   (labels ((render-column (line do-newline)
		      (pprint-vector stream line newline (1+ indent-size) (1+ dim-indicator))
		      (if do-newline
			  (if newline
			      (dotimes (k (1+ indent-size))
				(write-string " " stream)))))
		    (display-nth (n newline)
		      (render-column
		       (apply #'view tensor `(,@args ,n))
		       newline)))
	     ;; Displays first and last vector

	     ;; First vector
	     (dotimes (k midpoint)
	       (display-nth k T)
	       ;; Newline
	       (if newline
		   (progn
		     (when (= k (1- midpoint))
		       (write-char #\newline stream))
		     ;; Fix: the position of ... is collapsed.
		     (dotimes (_ (+ indent-size *matrix-element-displaying-size*))
		       (write-string " " stream))
		     (when (= k (1- midpoint))
		       (write-string "..." stream))
		     (write-char #\newline stream)
		     (dotimes (k (1+ indent-size))
		       (write-string " " stream)))))

	     ;; Last Vector
	     (loop with size = (nth dim-indicator (shape tensor))
	           for k downfrom midpoint to 1 do
		     (progn
		       (display-nth (- size k) NIL)
		       (when (not (= k 1))
			 (write-char #\newline stream)
			 (dotimes (i (1+ indent-size))
			   (write-string " " stream)))))
	     (write-string ")" stream)))))))



(defun render-tensor (tensor &key (indent 0))
  "The function reader-tensor renders :vec parts"
  (when (typep tensor 'Scalartensor)
    (return-from render-tensor
      (with-output-to-string (str)
	(dotimes (i indent) (princ " " str))
	(format str "~a" (tensor-vec tensor)))))

  
  (with-output-to-string (out)
    (let ((*matrix-element-displaying-size*
	    (+ 3 (loop for i fixnum upfrom 0 below (apply #'* (compute-visible-actual-shape tensor))
		       maximize (length (format nil "~a" (vref tensor i)))))))
      (pprint-vector out tensor t indent)
      out)))

