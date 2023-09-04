
(in-package :cl-waffe2/vm.nodes)

;; (TODO): Better Error Printing of Shape-Error
;; Memo: In Progress.

;; [TODO] Update the error msg
;;  V call or forward
;; (call (Composite-Name) X Y) -> ...
;;                        ^--- ...
;;                          ^--- ...
;;

(defstruct Number-of-Args
  "f(x, y) with (call f x)"
  first-state
  out-state
  previous-subscripts)

(defstruct (shape-error-message
	    (:conc-name msgof-))
  (position 0 :type fixnum)
  (place-me-last nil :type boolean)
  (in-short "" :type string)
  (content "" :type string)
  (suggestion "" :type string))

(defun describe-tensor (tensor)
  (declare (type AbstractTensor tensor))
  (format nil "~a{~a}~a"
	  (class-name (class-of tensor))
	  (dtype      tensor)
	  (shape tensor)))

(defun build-shape-error (forward-or-call model
			  where-decl
			  received predicted shape-error-list
			  &aux
			    (model-name (class-name (class-of model))))
  "
ShapeError Template:

Shaping-Error: Can't forward/call the Node/Composite because shapings are invaild.

At: [A] [B]
     ----
       |
      Here

Received: (call (MODEL ...) A(a) B(a) C(10))
                              L(1) |    |
                                   L(2) |
                                       (3)

Excepted: (call (MODEL ...) A(10 10) B(10 10))

<<Call Form Attempted>>

<<Definitions>>

<<RankError>> or <<Dimension Do Not Match?>> or <<Shapes undetermined>>

Suggestion"

  ;; Design:
  ;;
  ;; (A B C)
  ;;    ^ Set B = 1
  (declare (type (and keyword (member :forward :call)) forward-or-call)
	   (type symbol model-name)
	   (type (or AbstractNode Composite)))

  (with-output-to-string (out)
    (format out "[Shaping Error]:")

    (if (eql forward-or-call :call)
	(format out " The Composite ~a was called with invaild arguments." model-name)
	(case (checkpoint-state *shape-error-when*)
	  (:forward
	   (format out " The AbstractNode ~a was called with invaild arguments." model-name))
	  (:backward
	   (format out " The AbstractNode ~a inside backward definition was called with invaild arguments.

    (define-impl (~a (self ...)
        ...
        :backward ((self ...)
                    ...
                   (forward (~a ...) ...)
                              └── Detected when during the backward construction.
                   ...)))"
		   model-name (class-name (class-of (checkpoint-node-at *shape-error-when*)))
		   model-name))))

    (let* ((caller-name  (if (eql forward-or-call :forward) "forward" "call"))
	   (subject-name (if (eql forward-or-call :call) model-name
			     (if (eql (checkpoint-state *shape-error-when*) :backward)
				 (class-name (class-of (checkpoint-node-at *shape-error-when*)))
				 model-name)))
	   (parsed (multiple-value-list (parse-subscript where-decl)))
	   (in-args (car parsed)))

      (format out "~%~% The constraint:~%    ~a: ~a~%" subject-name where-decl)

      (if (find 'number-of-args shape-error-list :test #'subtypep :key #'class-of)
	  (progn
	    (format out "
Received:
    (~a
        (~a ...)~a)
              └── Received Too ~a Arguments.
  
Excepted:
    (~a (~a ...)~a)
"
		    caller-name model-name
		    (with-output-to-string (args)
		      (dolist (r received)
			(princ #\newline args)
			(princ "         " args)
			(princ (describe-tensor r) args)))
		    (if (>= (length received) (length in-args))
			"Many"
			"Few")
		    caller-name model-name
		    (with-output-to-string (args)
		      (dolist (n in-args)
			(princ " " args)
			(princ n args)))))
	  ;; More Errors
	  (let* ((more-errors (loop for er in shape-error-list
				    if (msgof-place-me-last er) collect er))
		 (shape-error-list (loop for er in shape-error-list
					 unless (msgof-place-me-last er) collect er))
		 (error-by-position (make-list (length received)))
		 (print-laters))
	    (dolist (err shape-error-list)
	      (push err (nth (msgof-position err) error-by-position)))
	    (format out "
Received:
    (~a
        (~a ...)~a
        )
~a
Excepted:
    (~a
        (~a ...)~a
        )
~a
Predicted outputs of ~a:  ~a
~a
The operation was:
~a
"
		    caller-name model-name
		    (with-output-to-string (args)
		      (loop for errors in error-by-position
			    for act in received
			    for nth fixnum upfrom 0
			    for in-name in in-args
			    if errors do
			      (princ #\Newline args)
			      (princ "         " args)
			      (princ (describe-tensor act) args)
			      ;; If errors are too many: move into another place
			      (if (> (length errors) 1)
				  (progn
				    (format args " ─ ~a: " in-name)
				    (push nth print-laters)
				    (dolist (err errors)
				      (princ (msgof-in-short err) args)
				      (princ " " args)))
				  (progn
				    (format args " ── ~a" (msgof-content (car errors)))))))
		    ;; print-laters
		    (if print-laters
			(with-output-to-string (tmp)			  
			  (dolist (nth print-laters)
			    (format tmp "~%~a:~%" (nth nth in-args))
			    (dolist (err (nth nth error-by-position))
			      (format tmp "    ─ ~a~%" (msgof-content err)))))
			"")
		    ;; Displays Predicted Outputs and suggestions If any
		    caller-name model-name
		    (with-output-to-string (args)
		      (loop for errors in error-by-position
			    for name in in-args
			    for act  in received
			    for nth upfrom 0
			    if errors do
			      (princ #\Newline args)
			      (princ "        " args)
			      (format args "~a~a" name (shape act))
			      (if (> (length errors) 1)
				  (format args " ─> ~a: " name)
				  (format args " ── ~a" (msgof-suggestion (car errors))))))
		    (if print-laters
			(with-output-to-string (tmp)
			  (dolist (nth print-laters)
			    (format tmp "~%~a:~%" (nth nth in-args))
			    (dolist (err (nth nth error-by-position))
			      (format tmp "    ─ ~a~%" (msgof-suggestion err)))))
			"")
		    model-name
		    predicted
		    (if more-errors
			(with-output-to-string (tmp)
			  (format tmp "More:~%")
			  (dolist (m more-errors)
			    (format tmp "~a~%" (msgof-content m))))
			"")
		    model))))))
