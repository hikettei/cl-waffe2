
(in-package :cl-waffe2/vm.nodes)

;; shaping-formatのパーサーなど

;; defstruct TransimissionState


;; (defstruct NodeState)

(defun bnf-parse-variables (exps &aux (pointer 0) (result))
  (flet ((parse-variable (exp)
	   exp))
    ;; [ -> symbol *n -> ]

    (let ((bracket-state '[)
	  (content-tmp))
      ;; bracket-state the symbol currently parsing
      ;; it has three state:
      ;; [ -> excepting [ for the coming next token
      ;; ] -> excepting ] or symbol for the coming next token
      ;;

      (dolist (exp exps)
	(cond
	  ((and (eql bracket-state '[)
		(eql exp '[))
	   (setq bracket-state ']))
	  ((and (eql bracket-state '[)
		(eql exp ']))
	   (error 'subscripts-content-error
		  :msg (format nil "The token anticipated to come is: [
However, got ~a.
When: ~a
At  : ~ath symbol, ~a"
			       exp
			       exps
			       pointer
			       (nth pointer exps))))
	  ((and (eql bracket-state '])
		(eql exp '[))
	   (error 'subscripts-content-error
		  :msg (format nil "The token anticipated to come is: ]
However, got ~a.
This is because of unmatched brackets or brackets are nested in the violation of the syntax rules.
When: ~a
At  : ~ath symbol, ~a"
			       exp
			       exps
			       pointer
			       (nth pointer exps))))
	  ((and (eql bracket-state '])
		(eql exp ']))
	   (setq bracket-state '[)
	   (push (parse-variable (reverse content-tmp)) result)
	   (setq content-tmp nil))
	  (T
	   (push exp content-tmp)))
	(incf pointer 1))
      (reverse result))))

(defun bnf-parse-let-phase (exps &aux (results))

  (let ((pointer 0)
	(var-name-tmp nil)
	(excepted-mode :variable-name)) ;; :symbol := :init-form
    (dolist (exp exps)
      (cond
	((and (eql excepted-mode :variable-name)
	      (and (symbolp exp) ;; i.e.: exp is variable name
		   (not (eql exp '=))))
	 (setq var-name-tmp exp)
	 (setq excepted-mode :=))
	((and (eql excepted-mode :=)
	      (eql exp '=))
	 (setq excepted-mode :init-form))
	((eql excepted-mode :init-form)
	 (setq excepted-mode :variable-name)
	 (push (list var-name-tmp exp) results))
	(T
	 (error 'subscripts-content-error
		:msg (format nil "Couldn't parse let-phase of subscripts.
Because the parser anticipated to come: ~a
But got: ~a.

Subscript: ~a
At       : ~ath Token, ~a"
			     excepted-mode
			     exp
			     exps
			     pointer
			     (nth pointer exps)))))
      (incf pointer 1))
    (reverse results)))

(defun bnf-parse-subscripts-toplevel (subscripts)
  "Toplevel of Subscripts

subscripts -> variables

Return:
(values forward-state out-state let-binding)"

  (macrolet ((with-fundamental-key ((key-place target-key ignorable) &body body)
	       `(let ((,key-place (position ',target-key subscripts)))
		  ;; Fundamental Keys: ->, and let can be used at once. (but let can be omitted)
		  
		  (when (> (count ',target-key subscripts) 1)
		    (error 'subscripts-format-Error
			   :because :too-many
			   :target ',target-key
			   :subscript subscripts))

		  (when (and (not ,ignorable)
			     (= 0 (count ',target-key subscripts)))
		    (error 'subscripts-format-error
			   :becuase :not-found
			   :target ',target-key
			   :subscript subscripts))
		  ,@body)))
    (with-fundamental-key (arrow -> nil)
      (with-fundamental-key (let-binding let t)
	
	(unless let-binding
	  (setq let-binding (length subscripts)))
	
	(let ((input-part (loop for i fixnum
				upfrom 0
				  below arrow
				collect (nth i subscripts)))
	      (output-part (loop for i fixnum
				 upfrom arrow
				   below let-binding
				 collect (nth i subscripts)))
	      (let-part    (loop for i fixnum
				 upfrom let-binding
				   below (length subscripts)
				 collect (nth i subscripts))))

	  (unless (eql (car output-part) '->)
	    (error 'subscripts-format-error
		   :because :invaild-template-order
		   :target '->
		   :subscript subscripts
		   :msg "Please follow this template:
 Input-Shape -> Output-Shape where X = 0
(the where phase is optional.)
(Perhaps this is because of: The phase let never comes before -> phase comes.)"))

	  (when (and (not (null let-part))
		     (not (eql (car let-part) 'let)))
	    (error 'subscripts-format-error
		   :because :invaild-template-order
		   :target 'let
		   :subscript subscripts
		   :msg "Please follow this template:
 Input-Shape -> Output-Shape where X = 0
(the where phase is optional.)"))

	  (values (bnf-parse-variables input-part)
		  (bnf-parse-variables (cdr output-part))
		  (bnf-parse-let-phase (cdr let-part))))))))

;; export
(defun parse-subscript (subscripts)
  "Subscripts are following format:

subscripts := variables -> variables

variables := variable
variables := variable variables

variable := [shape-subscripts]

条件式としてEvalできる形にする。
where [a[1] a[0]] [b[1] b[0]] -> [a[1] a[0]]
内部でさらに変数定義

where [a[0~t]] -> [a[x]] let x = shape (defnode引数を引用)

BASIC Format:

[Shape] [Shape] ... -> [Shape] [Shape] where x = 0, y = 1, ...

batch-size <- スコープが上位の変数も参照できるようにしたい。

[batch-size x[0] x[1]]"
  (declare (type list subscripts))

  ;; replace [a b] into [ a b ] (couldn't intepreted as a separated token)
  (multiple-value-bind (first-state out-state let-binding)
      (bnf-parse-subscripts-toplevel
       (read-from-string
	(regex-replace-all "\\]"
			   (regex-replace-all
			    "\\["
			    (format nil "~a" subscripts)
			    " [ ")
			   " ] ")))

    (print first-state)
    (print out-state)
    (print let-binding)
    ))


