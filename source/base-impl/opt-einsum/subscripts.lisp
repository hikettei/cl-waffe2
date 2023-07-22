
(in-package :cl-waffe2/base-impl)

;;
;; The file subscripts.lisp provides the parser of einsum syntax.
;;

;; I borrowed the parser from Subscript DSL
(defun parse-einsum-syntax (einsum)
  "Parses a given subscripts of einsum.
A [i j] -> B [i j]
  before      after

before = `((i j))
after =  `((i j))
before-symbols = (list A)
after-symbols  = (list A)"
  
  (multiple-value-bind (before after before-symbols after-symbols let-binding)
      (cl-waffe2/vm.nodes::parse-subscript einsum :fixed nil) ;; ~ isn't ignored

    ;; where a = 1 pharse can't be appeared in einsum syntax.
    
    ;;(when (not (null let-binding))
    ;;  (error "parse-einsum-syntax: `where` should not be used in einsum subscripts."))

    (when (not (= (length before) (length before-symbols)))
      (error "parse-einsum-syntax:
~a
  ^ All subscripts should be given their own name like: A[i j]" einsum))

    (when (not (= (length after-symbols) 1))
      (error "parse-einsum-syntax:
Regardless of implic/explict mode, (length after) should be one.
"))
    
    (values before after before-symbols after-symbols let-binding)))



;; ======================================================================================================
;;  [Parsing Einsum Syntax]

;; Example1: (einsum A[i j] B[j k] -> [i k])

;; => Create the corresponding data structures
;;
;; (dotimes (i 100)
;;   (dotimes (j 100)
;;      (dotimes (k 100)                            
;;         (setf (aref out i k)                     
;;               (* (aref x i j) (aref y j k))))))
;;
;; ↓ i.e.:
;; dotimes i ... 100, j ... 100                                    || Iteration Parts
;;   setf(->(out, 0, 0~100), ->(x, 0, 0), ->(y, 0, 0~100))         || Kernel    Parts (as i named so)
;;                         ...
;;   setf(->(out, i, 0~100), (* ->(x, i, j), ->(y, j, 0~100)))
;;

;; (einsum A[i i] -> B[i]) corresponds with:
;;
;; loop i = 0...n do
;;    (setf (aref B i) (+ (aref A i i)))
;;
;; ======================================================================================================

(defstruct (Subscript
	    (:constructor make-subscript
		(char
		 axis
		 nth-tensor)))
  (char char :type symbol)
  (axis axis :type fixnum)
  (nth-tensor nth-tensor :type fixnum))

(defun ->subscript (chars nth-tensor)
  (loop for c in chars
	for i upfrom 0
	collect (make-subscript c i nth-tensor)))


(defun determine-iterator-shapes (from to declared broadcast)
  (let ((syms (remove-duplicates (copy-list (alexandria:flatten `(,@from ,@to))))))
    (loop for sym in syms
	  if (and (not (find sym declared :test #'symbol-eq))
		  (not (symbol-eq sym broadcast)))
	    collect sym)))


(defun make-einsum (from-names to-names from-shapes to-shapes determined-symbols &aux (~ '~))
  "Creates the data structure above
~ = broadcastable"

  (let* ((from-shapes (loop for nth-tensor upfrom 0
			    for shape in from-shapes
			    collect (->subscript shape nth-tensor)))
	 (to-shapes   (loop for nth-tensor upfrom 0
			    for shape in to-shapes
			    collect (->subscript shape nth-tensor))))
    
    ))


