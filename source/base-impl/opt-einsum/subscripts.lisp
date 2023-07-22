
(in-package :cl-waffe2/base-impl)


;; ==========================================
;;  Subscript parsers
;; ==========================================


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


;;(defstruct (SSymbol
;;	    :constructor (make-ssymbol))
;; belongs-to nth ...)


