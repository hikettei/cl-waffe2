
;; Referenced: mgl-mat, melisgl.
;; https://github.com/melisgl/mgl-mat/blob/master/src/blas.lisp
;; https://github.com/melisgl/mgl-mat/blob/37e5d93f7b55d49b039ecd711bdb512994f607fb/src/util.lisp#

(in-package :cl-waffe2/backends.cpu)



(defun ctype-blas-prefix (ctype)
  ":float -> s, :dtype -> d"
  (ecase ctype
    ((nil) "")
    ((:float) "s")
    ((:double) "d")))


(defun blas-function-name (name ctype)
  (let ((*package* (find-package :cl-waffe2/backends.cpu)))
    (read-from-string (format nil "blas-~A~A" (ctype-blas-prefix ctype)
                              name))))

(defvar *mat-param-type*)

(defun param-name (param)
  (first param))

(defun param-direction (param)
  (or (third param) :input))

(defun param-type (param)
  (if (eq (second param) :mat)
      *mat-param-type*
      (second param)))

(defun mat-param-p (param)
  (eq (second param) :mat))

(defun non-mat-output-param-p (param)
  (and (not (mat-param-p param))
       (eq (param-direction param) :output)))

(defun ensure-pointer-param (param)
  (let ((ctype (param-type param)))
    (if (and (listp ctype)
             (eq (first ctype) :pointer))
        param
        (list (param-name param)
              `(:pointer ,ctype) (param-direction param)))))


(defun map-tree (fn tree)
  (let ((tree (funcall fn tree)))
    (if (listp tree)
        (mapcar (lambda (subtree)
                  (map-tree fn subtree))
                tree)
        tree)))

(defun convert-param-type (object)
  (cond ((eq object :float)
         :double)
        (t
         object)))

(defun convert-param-types (params type)
  (if (eq type :double)
      (map-tree #'convert-param-type params)
      params))


(defun blas-foreign-function-name (name ctype)
  (format nil "~A~A_" (ctype-blas-prefix ctype)
          (string-downcase (symbol-name name))))

(defun blas-funcall-form (name ctype params return-type args)
  (let ((cname (blas-foreign-function-name name ctype)))
    `(foreign-funcall
      ,cname
      ,@(loop for param in params
              for arg in args
              append (list (convert-param-types
                            (param-type param)
                            ctype)
                           arg))
      ,(convert-param-types return-type ctype))))

(defun blas-call-form* (params args fn)
  (if (endp params)
      (funcall fn (reverse args))
      (let* ((param (first params))
             (name  (param-name param))
             (ctype (param-type param))
             (direction (param-direction param)))
        (if (mat-param-p param)
            (let ((arg (gensym (symbol-name name))))
              `(let ((,arg ,name))
                 ,(blas-call-form* (rest params) (cons arg args) fn)))
            (if (and (listp ctype)
                     (eq (first ctype) :pointer))
                (let ((pointer-ctype (second ctype))
                      (arg (gensym (symbol-name name))))
                  `(with-foreign-object (,arg ,pointer-ctype)
                     ,@(when (member direction '(:input :io))
                         `((setf (mem-ref ,arg ,pointer-ctype) ,name)))
                     ,(blas-call-form* (rest params) (cons arg args) fn)
                     ,@(when (member direction '(:io :output))
                         `((mem-ref ,arg ,pointer-ctype)))))
                (blas-call-form* (rest params) (cons name args) fn))))))

(defun blas-call-form (blas-name ctype params return-type)
  (blas-call-form* params ()
                   (lambda (args)
                     (blas-funcall-form blas-name ctype
                                        params return-type args))))

(defmacro define-blas-function ((name &key (ctypes '(:float :double)))
                                (return-type (&rest params)))
  (let* ((*mat-param-type* '(:pointer :float))
         (params    (mapcar #'ensure-pointer-param params))
         (in-params (remove-if #'non-mat-output-param-p params))
         (lisp-parameters (mapcar #'param-name in-params)))
    
    `(progn
       ,@(loop for ctype in ctypes
               collect `(defun ,(blas-function-name name ctype)
                          ,lisp-parameters
                          ,(let ((params (convert-param-types params ctype)))
                             (blas-call-form name ctype params return-type)))))))

