
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
		:msg (format nil "Couldn't parse where-phase of subscripts.
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
      (with-fundamental-key (let-binding where t)
	
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
(Perhaps this is because of: The phase where never comes before -> phase comes.)"))

	  (when (and (not (null let-part))
		     (not (symbol-eq (car let-part) 'where)))
	    (error 'subscripts-format-error
		   :because :invaild-template-order
		   :target 'where
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

(defun build-subscript-error-note (&key
				     all-subscript
				     determined-shape
				     determined-out
				     symbol
				     excepted
				     but-got
				     nth-argument
				     target-shape)
  (format nil "
The Function is defined as :  ~a
Determined Shape           :  ~a -> ~a
Excepted: ~a = ~a
Butgot  : ~a = ~a
Because : The ~ath argument's shape is ~a.
"
	  all-subscript
	  determined-shape
	  determined-out
	  symbol
	  excepted
	  symbol
	  but-got
	  nth-argument
	  target-shape))

;; 二度とこのコード読みたくない
;; TODO: Fix: let -> where OK
;; TODO 四次元の時out-stateを構築できない
;; TODO: ~のShapeを判定
;; TOOD: out-stateのエラーのnote
;; TODO: 仕様を書く
;; TODO: Support it: where a = (1 2 3)
;; TODO: Optimize them... (After I've confirmed it works well)
(defun create-subscript-p (subscripts
			   &key
			     (macroexpand nil)
			   &aux
			     (previous-subscripts (gensym "PreviousShape"))
			     (undetermined-shape-tmp (gensym "UD"))
			     (all-conditions (gensym "Conds"))
			     (pos (gensym "pos")))
  (multiple-value-bind (first-state
			out-state
			let-binding)
      (parse-subscript subscripts)

    ;; [~ a b] [a b ~]
    ;; [~ a b] [~ k b] let k = (/ b 2)
    ;; boundp
    ;; TopLevelNode -> Shape決定
    ;; [a b c] [a b c] -> [a b c] let k = 1
    ;; let k = listを許す
    ;; Priority: 1. let-binding, defparameter 2. determined by cl-waffe
    ;; [x y z] let x = 1. In this case, x is 1. and y and z are 2.

    ;; TODO: Error Builder
    ;; TODO: [x ~ y] (1) <- regard it as error.

    ;; 1.
    ;; 2. (Other reasons that could be helpful)

    ;; [~ x y] [~ x y] -> [x y] with (5 1 2) (1 1 2) is OK
    ;; [~ x y] [~ x y] -> [~ x y] with (5 1 2) (1 1 2) is NG

    (let* ((least-required-dims (loop for s in first-state
				      collect (length (remove '~ s :test #'symbol-eq))))
	   ;; (1 2 3) for [x y] is invaild on the other hand (1 2 3) for [~ x y] is ok.
	   (max-required-dims (loop for s in first-state
				    if (find '~ s :test #'symbol-eq)
				      collect -1
				    else
				      collect (length s)))
	   (out-omit-p (find '~ (flatten out-state) :test #'symbol-eq))
	   ;; If ~ syntax inn't used in out-state, Anything is ok as ~.
	   ;; Unless then, ~ must be used as the same meaning in all args.
	   (common-symbols (get-common-symbols `(list ,first-state ,out-state)))
	   (body
	     `#'(lambda (,previous-subscripts &aux (,all-conditions))
		  ;; previous-suscriptsから次のSubscriptsを作成
		  ;; If any, return error condition
		  "Return: (values next-state condition)"
		  ;; TODO: Judge At-Least dims and return error.
		  ;; 1. Determine symbols, defined by let-binding

		  (unless (= (length ,previous-subscripts)
			     (length ',least-required-dims))
		    (push
		     (format nil "The function is declared as: ~a -> ~a but the given arguments was: ~a.~%Please assure that the number of arguments is ok." ',first-state ',out-state ,previous-subscripts)
		     ,all-conditions))

		  (mapc
		   #'(lambda (declared act)
		       (unless (<= declared (length act))
			 (push
			  ;; TODO: More infomation!
			  (format
			   nil
			   "The dimension must satisfy: dimensions >= ~a
Because the function is declared as: ~a -> ~a"
			   declared
			   ',first-state
			   ',out-state)
			  ,all-conditions)))
		   ',least-required-dims ,previous-subscripts)

		  (mapc
		   #'(lambda (declared act)
		       (unless (or (= declared -1)
				   (= declared (length act)))
			 (push
			  (format
			   nil
			   "The dimension must at least satisfy: dimension = ~a
Because the function is declared as: ~a -> ~a"
			   declared
			   ',first-state
			   ',out-state)
			  ,all-conditions)))
		   ',max-required-dims ,previous-subscripts)
		  
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
		    

		    ;; TODO: When let-binding includes list, use it directly.
		    
		    ;; Identify Non-Determined Symbols

		    ,@(loop for i fixnum upfrom 0
			    for subscript in first-state
			    collect
			    `(progn
			       ,(format nil "[MacroExpand] For: ~a" (nth i first-state))
			       ,@(loop with shape = `(nth ,i ,previous-subscripts)
				       with pos1 = (or (position '~ subscript :test #'symbol-eq) (length subscript)) ;; when [a b ~ c d] and (1 2 3 4 5). the index of b
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
						       (push
							(build-subscript-error-note
							 :all-subscript ',subscripts
							 :determined-shape (list ,@(map 'list #'(lambda (x) `(list ,@x)) first-state))
							 :determined-out (list ,@(map 'list #'(lambda (x) `(list ,@x)) out-state))
							 :symbol ',var
							 :excepted ,var
							 :but-got (nth ,nth-arg ,shape)
							 :nth-argument ,nth-arg
							 :target-shape ,shape)
							,all-conditions))
						     (setq ,var (nth ,nth-arg ,shape)))
						    ((> (- (length ,shape) ,nth-arg) ,pos1)
						     ;; here, we can detect errors
						     
						     (let ((,pos (- (1- (length ,shape)) (- ,pos2 ,nth-arg))))
						       (when (and (numberp ,var)
								  (not
								   (= ,var (nth ,pos ,shape))))
							 (push
							  (build-subscript-error-note
							   :all-subscript ',subscripts
							   :determined-shape (list ,@(map 'list #'(lambda (x) `(list ,@x)) first-state))
							   :determined-out (list ,@(map 'list #'(lambda (x) `(list ,@x)) out-state))
							   :symbol ',var
							   :excepted ,var
							   :but-got (nth ,nth-arg ,shape)
							   :nth-argument ,nth-arg
							   :target-shape ,shape)
							  ,all-conditions))
						       (setq ,var (nth ,pos ,shape)))))
				       else
					 collect `(loop for ,pos fixnum
							upfrom ,pos1
							  below (- (length ,shape) ,pos2)
							
							do (progn
							     (when (and ,out-omit-p
									(numberp (nth ,pos ,undetermined-shape-tmp))
									(not (= (nth ,pos ,undetermined-shape-tmp) (nth ,pos ,shape))))
							       ;; More details
							       (push
								(format
								 nil
								 "Couldn't idenfity ~~ ~~ is determined as ~a butgot: ~a" (nth ,pos ,undetermined-shape-tmp) (nth ,pos ,shape))
								,all-conditions))
							     (setf (nth ,pos ,undetermined-shape-tmp) (nth ,pos ,shape)))))))

		    (flet ((merge-and-determine (shapes)
			     (let ((out
				     (map
				      'list
				      #'(lambda (s)

					  ;; Use-priorities:
					  ;; 1. [a b]
					  ;; 2. ~

					  ;; TODO: Auto Padding
					  (or
					   (unless (symbolp s)
					     s)
					   (and
					    (push
					     (format
					      nil
					      "Couldn't Determine this symbol: ~a" s)
					     ,all-conditions)
					    '~)))
				      shapes)))
			       out)))
		      (let ((~ (remove '~ ,undetermined-shape-tmp :test #'symbol-eq)))
			(values
			 ;; (list out-shape1 out-shape...n)
			 (list ,@(map 'list #'(lambda (arg)
						`(flatten
						  (merge-and-determine
						   (list ,@arg))))
				      out-state))
			 (reverse ,all-conditions))))))))
      (when macroexpand
	(print body))
      (eval body))))

