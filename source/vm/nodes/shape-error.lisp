
(in-package :cl-waffe2/vm.nodes)

(defstruct Rank-Error
  "Rank Error"
  (position 0 :type fixnum)
  (excepted-rank 0 :type fixnum)
  (butgot 0 :type fixnum))

(defstruct Number-of-Args
  "f(x, y) with (call f x)"
  first-state
  out-state
  previous-subscripts)

(defstruct Rank-Atleast-error
  "A[~ i j] with (1)"
  position
  first-state
  declared
  butgot)

(defstruct Fixed-Arg-Rank-Error
  "A[i j] with (1 2 3)"
  position
  first-state
  declared
  butgot)

(deftype call-form-error-t ()
  `(or number-of-args))

(deftype rank-error-t ()
  "A family of Rank-Error struct"
  `(or rank-error
       rank-atleast-error
       fixed-arg-rank-error))

(defun compile-call-form-error (nth errorlist)
  (declare (type list errorlist))
  (with-output-to-string (result)
    (format result "== ~:R, calling form is invaild. ==~%" nth)
    (let ((target (car errorlist)))
      (when (typep target 'call-form-error-t)
	(let ((nargs  (length (number-of-args-first-state target)))
	      (butgot (length (number-of-args-previous-subscripts target))))
	  (format result "~%Since the node is declared as: ~a -> ~a, the number of arguments must correspond.

It is too ~a: ~a"

		(number-of-args-first-state target)
		(number-of-args-out-state   target)
		(if (> butgot nargs)
		    "many"
		    "few")
		(number-of-args-previous-subscripts target)))))))
  
(defun compile-rank-error (nth errorlist)
  (declare (type list errorlist))
  (with-output-to-string (result)
    (format result "== ~:R, rank error is detected. ==~%" nth)
    
    ))

(defun build-shape-error (shape-error-list)
  "
ShapeError Template:

[shape-error] Shape Error was detected when doing ...

<<Call Form Attempted>>

<<Definitions>>

<<RankError>> or <<Dimension Do Not Match?>> or <<Shapes undetermined>>"

  ;; Design:
  ;;
  ;; (A B C)
  ;;    ^ Set B = 1
  )

