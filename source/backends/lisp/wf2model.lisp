
(in-package :cl-waffe2/backends.lisp)


(define-model-format (:wf2model cl-waffe2/backends.lisp:LispTensor)
		     :save-weights ((path state-dict)
				    (waffe2-format-saveas path state-dict))
		     :load-weights ((path)
				    (waffe2-format-loadas path)))

(defun waffe2-format-saveas (path state-dict)
  (let ((base-dir (pathname (format nil "~a/" path)))
	(config-place (pathname (format nil "~a/parameters.json" path))))
    (flet ((make-path (key)
	     (format nil "~a/~a.npy" path key)))
      (ensure-directories-exist base-dir)
      (with-open-file (stream config-place :direction :output :if-exists :supersede :if-does-not-exist :create)
	(jojo:with-output (stream)
	  (jojo:with-object
	    (maphash
	     #'(lambda (key value)
		 (if (typep value 'AbstractTensor)
		     (progn
		       (numpy-file-format:store-array
			(tensor-vec
			 (cl-waffe2:device-as
			  value
			  'cl-waffe2/backends.lisp:LispTensor))
			(make-path key))
		       (jojo:write-key-value key (make-path key)))
		     (progn
		       (jojo:write-key-value key value))))
	     (state-dict-table state-dict))))))
    T))

(defun waffe2-format-loadas (path)
  (let ((state-dict-config
	  (jojo:parse
	   (uiop:read-file-string
	    (format nil "~a/parameters.json" path))
	   :as :hash-table))
	(result (make-hash-table :test #'equal)))
    (maphash
     #'(lambda (k v)
	 (if (stringp v)
	     (setf (gethash k result)
		   (cl-waffe2:device-as
		    (cl-waffe2:change-facet
		     (numpy-file-format:load-array v)
		     :direction 'AbstractTensor)
		    (car *using-backend*)))
	     (setf (gethash k result) v)))
     state-dict-config)
    (from-state-dict result)))

