
(in-package :cl-waffe2/vm.generic-tensor)


(define-condition Indexing-Error (simple-error)
  ((content :initarg :content))
  (:documentation "")
  (:report
   (lambda (c s)
     (format s "[cl-waffe] Indexing-Error: ~a" (slot-value c 'content)))))

(defmacro indexing-error (content &rest args)
  `(error (make-condition 'indexing-error
			  :content (format nil ,content ,@args))))


(define-condition View-Indexing-Error (indexing-error)
  ((content :initarg :content))
  (:documentation "")
  (:report
   (lambda (c s)
     (format s "[cl-waffe] View-Indexing-Error: ~a" (slot-value c 'content)))))

(defmacro view-indexing-error (content &rest args)
  `(error (make-condition 'view-indexing-error
			  :content (format nil ,content ,@args))))

(define-condition Shaping-Error (indexing-error)
  ((content :initarg :content))
  (:documentation "")
  (:report
   (lambda (c s)
     (format s "[cl-waffe] Shaping-Error: ~a" (slot-value c 'content)))))

(defmacro shaping-error (content &rest args)
  `(error (make-condition 'Shaping-error
			  :content (format nil ,content ,@args))))

(define-condition node-compile-error (simple-error)
  ((content :initarg :content))
  (:report
   (lambda (c s)
     (format s "[cl-waffe2] Node-Compile-Error: ~a" (slot-value c 'content)))))

(defmacro node-compile-error (content &rest args)
  `(error (make-condition 'node-compile-error :content (format nil ,content ,@args))))
