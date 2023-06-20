
(in-package :cl-user)

(defpackage :cl-waffe2.docs
  (:use
   :cl
   :cl-waffe2
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes
   :cl-waffe2/distributions
   :cl-waffe2/base-impl
   :cl-waffe2/backends.lisp
   :cl-waffe2/backends.cpu
   :cl-waffe2/nn
   :cl-waffe2/optimizers
   :cl-ppcre)
  (:export
   #:generate))

(in-package :cl-waffe2.docs)

;; Utils for genearting documents

(defmacro with-page (title-binding-symbol
		     title-name
		     &body
		       body)
  `(setq ,title-binding-symbol
	 (with-section ,title-name
	   ,@body)))

(defmacro with-section (title-name
			&body body
			&aux (output-to (gensym)))
  `(with-output-to-string (,output-to)
     (format ,output-to "~%@begin(section)~%@title(~a)" ,title-name)
     (macrolet ((insert (content &rest args)
		  `(format ,',output-to "~%~a" (format nil ,content ,@args)))
		(b (content &rest args)
		  `(format ,',output-to "~%@b(~a)" (format nil ,content ,@args)))
		(image (url)
		  `(princ (format nil "~%@image[src=\"~a\"]()" ,url) ,',output-to))
		(url (url name)
		  `(princ (format nil "~%@link[uri=\"~a\"](~a)" ,url ,name) ,',output-to))
		(def (content)
		  `(format ,',output-to "~%@begin(def)~%~a~%@end(def)" ,content))
		(term (content)
		  `(format ,',output-to "~%@begin(term)~%~a~%@end(term)" ,content))
		(placedoc (package type name)
		  `(format ,',output-to "~%@cl:with-package[name=\"~a\"](~%@cl:doc(~a ~a)~%)~%" ,package ,type ,name))
		(item (content)
		  `(format ,',output-to "~%@item(~a)" ,content)))
       (macrolet ((with-section (title-name &body body)
		    `(progn
		       (format ,',output-to "~%@begin(section)~%@title(~a)~%" ,title-name)
		       ,@body
		       (format ,',output-to "~%@end(section)")))
		  (with-term (&body body)
		    `(progn
		       (format ,',output-to "~%@begin(term)")
		       ,@body
		       (format ,',output-to "~%@end(term)")))
		  (with-def (&body body)
		    `(progn
		       (format ,',output-to "~%@begin(def)")
		       ,@body
		       (format ,',output-to "~%@end(def)")))
		  (with-enum (&body body)
		    `(progn
		       (format ,',output-to "~%@begin(enum)")
		       ,@body
		       (format ,',output-to "~%@end(enum)")))
		  (with-deflist (&body body)
		    `(progn
		       (format ,',output-to "~%@begin(deflist)")
		       ,@body
		       (format ,',output-to "~%@end(deflist)")))
		  (with-lisp-code (content)
		    `(format ,',output-to "~%@begin[lang=lisp](code)~%~a~%@end[lang=lisp](code)" ,content))
		  (with-shell-code (content)
		    `(format ,',output-to "~%@begin[lang=shell](code)~%~a~%@end[lang=shell](code)" ,content))
		  (with-api (type function-name &body body)
		    `(progn
		       (format ,',output-to "~%@begin(section)~%@title(~a)~%" ,function-name)
		       (placedoc "cl-waffe" ,type ,function-name)
		       ,@body
		       (format ,',output-to "~%@end(section)")
		       ))
		  (with-eval (code)
		    `(progn ; `(progn (read-from-string ~))
		       (format ,',output-to "~%~%~%@b(Input)")
		       (format ,',output-to "~%@begin[lang=lisp](code)~%CL-WAFFE>~a~%@end[lang=lisp](code)~%" ,code)
		       (format ,',output-to "~%@b(Output)~%")
		       (format ,',output-to "~%@begin[lang=lisp](code)~%~a~%@end[lang=lisp](code)~%" (eval (read-from-string ,code)))))
		  (with-evals (&rest codes)
		    `(progn
		       (format ,',output-to "~%~%~%")
		       (format ,',output-to "@b(REPL:)")
		       (dolist (code ',codes)
			 (format ,',output-to "~%@begin[lang=lisp](code)~%CL-WAFFE> ~a~%@end[lang=lisp](code)~%" code)
			 (format ,',output-to "~%@begin[lang=lisp](code)~%~a~%@end[lang=lisp](code)~%" (eval (read-from-string code))))))
		  )
	 ,@body))
     (format ,output-to "~%@end(section)")))


(defparameter *target-dir* "./docs/scriba")

(defun write-scr (filepath content)
  (with-open-file (str (out-dir filepath)
                       :direction :output
                       :if-exists :supersede
                       :if-does-not-exist :create)
    (format str "~a" content)))

(defun out-dir (name)
  (format nil "~a/~a.scr" *target-dir* name))


(defparameter *overview* "")
(defparameter *setup* "")
(defparameter *concept* "")

(defparameter *distributions* "")

(defun generate ()
  (write-scr "overview" *overview*)
  (write-scr "distributions" *distributions*)
  
  )
