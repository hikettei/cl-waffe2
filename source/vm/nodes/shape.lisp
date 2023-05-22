
(in-package :cl-waffe2/vm.nodes)

;; shaping-formatのパーサーなど

;; defstruct TransimissionState


;; (defstruct NodeState)

(defun symbol-eq (x y)
  (and
   (symbolp x)
   (symbolp y)
   (equal (symbol-name x)
	  (symbol-name y))))

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
	  ((and (symbol-eq bracket-state '[)
		(symbol-eq exp '[))
	   (setq bracket-state ']))
	  ((and (symbol-eq bracket-state '[)
		(symbol-eq exp ']))
	   (error 'subscripts-content-error
		  :msg (format nil "The token anticipated to come is: [
However, got ~a.
When: ~a
At  : ~ath symbol, ~a"
			       exp
			       exps
			       pointer
			       (nth pointer exps))))
	  ((and (symbol-eq bracket-state '])
		(symbol-eq exp '[))
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
	  ((and (symbol-eq bracket-state '])
		(symbol-eq exp ']))
	   (setq bracket-state '[)
	   (push (parse-variable (reverse content-tmp)) result)
	   (setq content-tmp nil))
	  (T
	   (push exp content-tmp)))
	(incf pointer 1))
      
      (when content-tmp
	(error 'subscripts-content-error
	       :msg (format nil "The rest tokens doesn't fit in no any syntax rule.: ~a" (reverse content-tmp))))
      
      (reverse result))))

(defun bnf-parse-let-phase (exps &aux (results))

  (let ((pointer 0)
	(var-name-tmp nil)
	(excepted-mode :variable-name)) ;; :symbol := :init-form
    (dolist (exp exps)
      (cond
	((and (symbol-eq excepted-mode :variable-name)
	      (and (symbolp exp) ;; i.e.: exp is variable name
		   (not (symbol-eq exp '=))))
	 (setq var-name-tmp exp)
	 (setq excepted-mode :=))
	((and (symbol-eq excepted-mode :=)
	      (symbol-eq exp '=))
	 (setq excepted-mode :init-form))
	((symbol-eq excepted-mode :init-form)
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
	       `(let ((,key-place (position ',target-key subscripts :test #'symbol-eq)))
		  ;; Fundamental Keys: ->, and let can be used at once. (but let can be omitted)
		  
		  (when (> (count ',target-key subscripts :test #'symbol-eq) 1)
		    (error 'subscripts-format-Error
			   :because :too-many
			   :target ',target-key
			   :subscript subscripts
			   :msg ""))

		  (when (and (not ,ignorable)
			     (= 0 (count ',target-key subscripts :test #'symbol-eq)))
		    (error 'subscripts-format-error
			   :because :not-found
			   :target ',target-key
			   :subscript subscripts
			   :msg ""))
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

	  (unless (symbol-eq (car output-part) '->)
	    (error 'subscripts-format-error
		   :because :invaild-template-order
		   :target '->
		   :subscript subscripts
		   :msg "Please follow this template:
 Input-Shape -> Output-Shape where X = 0
(the where phase is optional.)
(Perhaps this is because of: The phase let never comes before -> phase comes.)"))

	  (when (and (not (null let-part))
		     (not (symbol-eq (car let-part) 'let)))
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

[~ i j] -> [~ j i] let x = 1

[~] -> [~]

x -> [x y] let x = (list 1 2 3)

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
    (values first-state out-state let-binding)))

(defun get-common-symbols (symbols)
  (remove-duplicates (flatten symbols) :test #'symbol-eq))

(defun subscript-compatible-p (subscript-1 shape)

  )

(defun padding-subscript (subscript shape)
  "(a ~ b) (1 2 3 4) -> (a ~ ~ b)"
  (declare (type list subscript shape))
  (let* ((result (loop for s in shape collect '~))
	 (pos1   (or (position '~ subscript :test #'symbol-eq) 0))
	 (pos2   (or (position '~ (reverse subscript) :test #'symbol-eq) (length result))))
    ;; [a b ~ c] (3 3 3 3 3) pos1 = 1, pos2 = 0 
    (loop for i fixnum
	  upfrom 0
	    below pos1
	  do (setf (nth i result) (nth i shape)))
    
    (loop for i fixnum
	  downfrom (1- (length result))
	    to (- (length result) pos2)
	  do (setf (nth i result) (nth i shape)))
    result))

;; TODO: Fix: let -> where
(defun create-subscript-p (subscripts
			   &aux
			     (previous-subscripts (gensym "PreviousShape"))
			     (undetermined-shape-tmp (gensym "UD"))
			     (all-conditions (gensym "Conds")))
  (multiple-value-bind (first-state
			out-state
			let-binding)
      (parse-subscript subscripts)

    ;; TODO: ~ can be used at once
    ;; ~に遭遇: 残りのsubscript or prev-stateがn or 0になるまでpop
    ;; [~ a b] [a b ~]
    ;; [~ a b] [~ k b] let k = (/ b 2)
    ;; boundp
    ;; TopLevelNode -> Shape決定
    ;; [a b c] [a b c] -> [a b c] let k = 1
    ;; let k = listを許す
    ;; Priority: 1. let-binding, defparameter 2. determined by cl-waffe
    ;; [x y z] let x = 1. In this case, x is 1. and y and z are 2.

    (let* ((common-symbols (get-common-symbols first-state))
	   (body
	     `#'(lambda (,previous-subscripts &aux (,all-conditions))
		  ;; previous-suscriptsから次のSubscriptsを作成
		  ;; If any, return error condition
		  "Return: (values next-state condition)"
		  ;; TODO: Judge At-Least dims and return error.
		  ;; 1. Determine symbols, defined by let-binding
		  (let* (,@(map 'list #'(lambda (x)
					  `(,x ',x))
				common-symbols)
			 ,@let-binding
			 (,undetermined-shape-tmp
			   (loop for s
				 upfrom 0
				 below
				 (loop for p in ,previous-subscripts
				       maximize (length p))
				 collect '~)))
		    (declare (ignorable ~))
		    
		    ;; Identify Non-Determined Symbols

		    ,@(loop for i fixnum upfrom 0
			    for subscript in first-state
			    collect
			    `(progn
			       ,(format nil "[MacroExpand] For: ~a" (nth i first-state))
			       ,@(loop with shape = `(nth ,i ,previous-subscripts)
				       with pos1 = (or (position '~ subscript :test #'symbol-eq) 0) ;; when [a b ~ c d] and (1 2 3 4 5). the index of b
				       with pos2 = (or (position '~ (reverse subscript) :test #'symbol-eq) 0) ;; when [a b ~ c d] and (1 2 3 4 5), the index of c
				       
				       for nth-arg fixnum upfrom 0
				       for var in subscript
				       unless (symbol-eq var '~)
					 collect `(cond
						    ((< ,nth-arg ,pos1)
						     ;; here, we can detect errors
						     ;; ,var is determined and determined shapes doesn't match.
						     (when (and (numberp ,var)
								(not
								 (= ,var (nth ,nth-arg ,shape))))
						       (print "shape error detected")
						       (print ',var)
						       )
						     (setq ,var (nth ,nth-arg ,shape)))
						    ((> (- (length ,shape) ,nth-arg) ,pos1)
						     ;; here, we can detect errors
						     
						     (let ((pos (- (1- (length ,shape)) (- ,pos2 ,nth-arg))))
						       (when (and (numberp ,var)
								  (not
								   (= ,var (nth pos ,shape))))
							 (print "shape error detected")
							 (print ',var)
							 (print ,var)
							 )
						       (setq ,var (nth pos ,shape))))))))
		    ;; Determine ~
		    ,@(loop for i fixnum upfrom 0
			    for arg in first-state
			    collect
			    `(let ((out (padding-subscript (flatten (list ,@arg)) (nth ,i ,previous-subscripts))))
			       (loop for axis fixnum upfrom 0
				     for sym in ',arg
				     for res in out
				     for n in (nth ,i ,previous-subscripts)
				     if (symbol-eq res '~)
				       ;; here could we find shape-error.
				       do (progn
					    (if (and
						 (numberp (nth axis ,undetermined-shape-tmp))
						 (not (= (nth axis ,undetermined-shape-tmp) n)))
						
						(print "error")) ;; wakariyasuku
					    (setf (nth axis ,undetermined-shape-tmp) n)))))
		    ;; 数値とシンボルで返す
		    ;; 最適化のために、なるべくスカラー値としてShapeを特定

		    ,@(map 'list #'(lambda (arg)
				     `(print (list ,@arg)))
			   first-state)
		    (print ,undetermined-shape-tmp)
		    
		    ))))
      (print body)
      (eval body))))

