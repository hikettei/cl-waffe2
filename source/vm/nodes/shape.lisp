
(in-package :cl-waffe2/vm.nodes)

(defun range (start end)
  (loop for i upfrom start below end collect i))

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
At  : ~a symbol, ~a"
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

(defun separate-shape-and-name (list)
  "A[~] -> A and [~]"
  (let ((names-found)
	(parsed)
	(state :symbol))
    (loop for token in list
	  for p upfrom 0
	  do (cond
	       ((and (eql state :symbol)
		     (not (symbol-eq token '[)))
		(push token names-found)
		(setq state :[))
	       ((and (eql state :symbol)
		     (symbol-eq token '[))
		(push token parsed)
		(setq state :content))
	       ((eql state :[)
		(push token parsed)
		(setq state :content))
	       ((eql state :content)
		(push token parsed)
		(if (symbol-eq token '])
		    (setq state :symbol)))))
    
    (values (reverse names-found) (reverse parsed))))

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

	  (multiple-value-bind (input-vars input-part) (separate-shape-and-name input-part)
	    (multiple-value-bind (output-vars output-part) (separate-shape-and-name (cdr output-part))

	      (values input-vars
		      output-vars
		      (bnf-parse-variables input-part)
		      (bnf-parse-variables output-part)
		      (bnf-parse-let-phase (cdr let-part))))))))))

(defun preprocess-list (subscripts)
  (read-from-string
   (regex-replace-all "\\]"
		      (regex-replace-all
		       "\\["
		       (format nil "~a" subscripts)
		       " [ ")
		      " ] ")))

;; export
(defun parse-subscript (subscripts &key (fixed nil))
  "Subscripts are following format:

subscripts := variables -> variables

variables := variable
variables := variable variables

variable := [shape-subscripts]

where [a[0~t]] -> [a[x]] let x = shape (defnode引数を引用)

BASIC Format:

[Shape] [Shape] ... -> [Shape] [Shape] where x = 0, y = 1, ...

[~ i j] -> [~ j i] let x = 1

[~] -> [~]

x -> [x y] let x = (list 1 2 3)

[batch-size x[0] x[1]]

If fixed=t, ~ is ignored."
  (declare (type list subscripts))

  ;; replace [a b] into [ a b ] (couldn't intepreted as a separated token)
  (multiple-value-bind (inputs outputs first-state out-state let-binding)
      (bnf-parse-subscripts-toplevel
       (let ((out (preprocess-list subscripts)))
	 (if fixed
	     (loop for s in out
		   unless (symbol-eq s '~)
		     collect s)
	     out)))
    (values inputs outputs first-state out-state let-binding)))

(defun parse-transform-syntax (subscripts)
  "
The same as subscript dsl but the number of -> is inf.

Return: (values names subscripts where)
"
  (declare (type list subscripts))
  (let ((subscripts (preprocess-list subscripts))
	(separated-names)
	(parsed-subscripts)
	(let-bindings))
    (loop named iter
          while subscripts
	  do (let ((pos (or (position '-> subscripts :test #'symbol-eq)
			    (position 'where subscripts :test #'symbol-eq)
			    (length subscripts))))
	       (when (null pos)
		 (return-from iter))

	       (let ((target (subseq subscripts 0 pos)))
		 (if (find '= target :test #'symbol-eq)
		     (mapc #'(lambda (x) (push x let-bindings)) (bnf-parse-let-phase target))
		     (multiple-value-bind (name val) (separate-shape-and-name target)
		       (push name separated-names)
		       (push (bnf-parse-variables val) parsed-subscripts))))
	       (dotimes (i pos) (pop subscripts))
	       (pop subscripts)))
    
    (values (reverse separated-names)
	    (reverse parsed-subscripts)
	    (reverse let-bindings))))

;; (print (parse-transform-syntax `(A[i j] -> B[j k] -> C[k i] where k = 1)))

;; Bug: [~] A[~] -> B [~]
(defun where->backward (subscripts)
  "forward -> backward where ... => backward -> forward where ..."
  (multiple-value-bind (first-names output-names first-states out-states let-binding) (parse-subscript subscripts)

    `(,@(loop for i fixnum upfrom 0 below (length out-states)
	      if (nth i output-names)
		append `(,(nth i output-names) [ ,@(nth i out-states) ])
	      else
		append `([ ,@(nth i out-states) ]))
      ->
      ,@(loop for i fixnum upfrom 0 below (length first-states)
	      if (nth i first-names)
		append `(,(nth i first-names) [ ,@(nth i first-states) ])
	      else
		append `([ ,@(nth i first-states) ]))

      ,@(when let-binding
	  `(where
	    ,@(loop for let in let-binding
		    append `(,(car let) = ,(second let))))))))

(defun get-common-symbols (symbols)
  (remove-duplicates (flatten symbols) :test #'symbol-eq))

(defun build-subscript-error-note (&key
				     all-subscript
				     determined-shape
				     determined-out
				     symbol
				     excepted
				     but-got
				     nth-argument
				     target-shape)
  (format nil "Inconsistency of subscripts.
The Function is defined as :  ~a
Determined Shape           :  ~a -> ~a
Excepted: ~a = ~a
Butgot  : ~a = ~a
Because : The actual ~ath argument given has a shape of ~a.
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

(defun find-symbols (list)
  (loop for l in list
	if (symbolp l)
	  collect l))

(defmacro num-major-setq (var obj)
  `(setq ,var
	 (or (when (or (numberp ,var)
		       (listp   ,var))
	       ,var)
	     ,obj)))

(defmacro num-major-setf (var obj)
  `(setf ,var
	 (or (when (or (numberp ,var)
		       (listp ,var))
	       ,var)
	     ,obj)))


(defun shape-equal-1 (pos list1 list2)
  (and (= (length list1)
	  (length list2))
       (shape-equal (nth pos list1) (nth pos list2))))

(defun replace-nil (list)
  (map 'list #'(lambda (l) (if l l '?)) list))

;; 二度とこのコード読みたくない
;; Fix: subscript = [shape] where shape = `(1 2 3)
;; => ndimension doesn't match. (Currently avoided by using ~)
;; signifcantly slow compilation...
;; Optimize Me...
;; TODO: call it when defnode is called
;; Rewrite this program...
;; TODO: Rewrite this ugly code, and print better error-notes
(defun create-subscript-p (subscripts
			   &key
			     (allow-symbol nil)
			     (macroexpand nil)
			     (fixed nil)
			     (return-body nil)
			     (local-variables nil)
			   &aux
			     (previous-subscripts (gensym "PreviousShape"))
			     (undetermined-shape-tmp (gensym "UD"))
			     (all-conditions (gensym "Conds"))
			     (pos (gensym "pos"))
			     (undetermined-symbols (gensym "SYM"))
			     (rank-error-p (gensym "rerr")))
  "Creates a subscript-p function.

allow-symbol ... Returned shape can contain symbols.

local-variables ... symbols to be implictly added to `where`.
Example:
 (create-subscript-p `([x y] -> [x y]))
 (funcall * `((1 1))) -> (values `((1 1)) NIL)
"
  (declare (type list local-variables))
  (multiple-value-bind (input-names
			output-names
			first-state
			out-state
			let-binding)
      (parse-subscript subscripts :fixed fixed)
    (declare (optimize (speed 3)))
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

    ;; arguments with ~, has the equivalent number of dimension.

    (let* ((least-required-dims (loop for s in first-state
				      collect (length (the list (remove '~ s :test #'symbol-eq)))))
	   ;; (1 2 3) for [x y] is invaild on the other hand (1 2 3) for [~ x y] is ok.
	   (max-required-dims (loop for s in first-state
				    if (find '~ (the list s) :test #'symbol-eq)
				      collect -1
				    else
				      collect (length (the list s))))
	   (out-omit-p (if (find '~ (the list (flatten out-state)) :test #'symbol-eq)
			   t
			   nil))
	   (~symbol (or (find '~
			      (the list (flatten (list first-state)))
			      :test #'symbol-eq)
			'~))
	   (flexdim-n (loop for input in first-state
			    for k fixnum upfrom 0
			    if (find '~ (the list (flatten input)) :test #'symbol-eq)
			      collect k))
	   ;; If ~ syntax isn't used in out-state, Anything is ok as ~.
	   ;; Unless then, ~ must be used as the same meaning in all args.
	   (common-symbols (get-common-symbols (list first-state out-state)))

	   (body
	     `(lambda (,previous-subscripts
		       &aux
			 (,all-conditions)
			 (,rank-error-p nil)
			 (,undetermined-symbols (find-symbols (flatten ,previous-subscripts))))
		(declare (optimize (speed 1) (compilation-speed 3))
			 #+sbcl(sb-ext:muffle-conditions cl:style-warning sb-ext:compiler-note)
			 )
		;; previous-suscriptsから次のSubscriptsを作成
		;; If any, return error condition
		;; "Return: (values next-state condition)"
		;; TODO: Judge At-Least dims and return error.
		;; 1. Determine symbols, defined by let-binding
		

		(unless (= (length ,previous-subscripts)
			   (length ',least-required-dims))
		  (push
		   (format nil "The function is declared as: ~a -> ~a
but the actual argument given was: ~a.
=> Check that the number of arguments is correct." ',first-state ',out-state ,previous-subscripts)
		   ,all-conditions))

		,(when flexdim-n
		   `(let ((dim)
			  (accd))
		      (loop for k in ',flexdim-n
			    if (null dim)
			      do (setq dim (length (nth k ,previous-subscripts)))
			         (setq accd k)
			    unless (= dim (length (nth k ,previous-subscripts)))
			      do (push
				  (format nil "The dimensions of the ~ath input do not match.
Excepted Dimensions -> ~a (in accordance with ~ath input)
Butgot              -> ~a (shape=~a)
=> Consider using the function !flexible which enables broadcasting."
					  k
					  dim
					  accd
					  (length (nth k ,previous-subscripts))
					  (nth k ,previous-subscripts))
				  ,all-conditions))))
			  

		;; The Number of dimensions error.
		(mapc
		 #'(lambda (nth-arg declared act)
		     (unless (<= declared (length act))
		       (and
			(setq ,rank-error-p t)
			(push
			 (format
			  nil
			  "The number of dimensions must satisfy: dimensions >= ~a
Because the function is declared as: ~a -> ~a.
=> However, the actual ~ath argument given was ~a."
			  declared
			  ',first-state
			  ',out-state
			  (1+ nth-arg)
			  act)
			 ,all-conditions))))
		 (range 0 (length ,previous-subscripts))
		 ',least-required-dims
		 ,previous-subscripts)

		(mapc
		 #'(lambda (nth-arg declared act)
		     (unless (or (= declared -1)
				 (= declared (length act)))
		       (and
			(setq ,rank-error-p t)
			(push
			 (format
			  nil
			  "The ~ath argument is declared as: ~a
Accordingly, the argument must satisfy: dimensions = ~a
=> However, the actual ~ath argument given was ~a"
			  (1+ nth-arg)
			  (nth nth-arg ',first-state)
			  declared
			  (1+ nth-arg)
			  act)
			 ,all-conditions))))
		 
		 (range 0 (length ,previous-subscripts))
		 ',max-required-dims
		 ,previous-subscripts)

		;; Initializing Tables
		(let* (,@(map 'list #'(lambda (x)
					;; If the symbol is declared as localvar
					;; Use it as initial value
					(if (find x local-variables :test #'symbol-eq)
					    `(,x ,x)
					    `(,x)))
			      common-symbols) ;; Tables: 'a 'b 'c ...
		       ;;,@let-binding ;; initial element to 'a 'b 'c...
		       ,@(loop for let in let-binding
			       collect `(,(car let) ',(car let)))
		       ;; ~ = `(nil nil nil) at first.
		       (,undetermined-shape-tmp
			 (loop for s
			       upfrom 0
				 below
				 (loop for p in ,previous-subscripts
				       maximize (length p))
			       collect nil)))
		  (declare (ignorable ,@common-symbols))
		  
		  ;; TODO: When let-binding includes list, use it directly.
		  
		  ;; Identify Non-Determined Symbols

		  ,@(loop for i fixnum upfrom 0
			  for subscript in first-state
			  collect
			  `(progn
			     ,(format nil "[DEBUG] For: ~a" (nth i first-state))
			     ,@(loop with shape = `(nth ,i ,previous-subscripts)
				     with pos1 = (or (position '~ (the list subscript) :test #'symbol-eq) (length subscript)) ;; when [a b ~ c d] and (1 2 3 4 5). the index of b
				     with pos2 = (or (position '~ (reverse subscript) :test #'symbol-eq) 0) ;; when [a b ~ c d] and (1 2 3 4 5), the index of c
				     
				     for nth-arg fixnum upfrom 0
				     for var in subscript
				     unless (symbol-eq var '~)
				       collect `(cond
						  ((< ,nth-arg ,pos1)
						   ;; here, we can detect errors
						   ;; ,var is determined and determined shapes doesn't match.
						   (when (and (not (null ,var))
							      (or
							       ;; ('a 'a) or (10 10), not ('a 1) (1 'a)
							       (and (symbolp ,var)
								    (symbolp (nth ,nth-arg ,shape)))
							       (and (numberp ,var)
								    (numberp (nth ,nth-arg ,shape))))
							      (not
							       (equal ,var (nth ,nth-arg ,shape))))
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
						   (num-major-setq ,var (nth ,nth-arg ,shape)))
						  ((> (- (length ,shape) ,nth-arg) ,pos1)
						   ;; here, we can detect errors
						   
						   (let ((,pos (- (1- (length ,shape)) (- ,pos2 ,nth-arg))))
						     (when (and (not (null ,var))
								(or
								 ;; ('a 'a) or (10 10), not ('a 1) (1 'a)
								 (and (symbolp ,var)
								      (symbolp (nth ,nth-arg ,shape)))
								 (and (numberp ,var)
								      (numberp (nth ,nth-arg ,shape))))
								(not
								 (equal ,var (nth ,pos ,shape))))
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
						     (num-major-setq ,var (nth ,pos ,shape)))))
				     else
				       collect
				     `(loop for ,pos fixnum
					    upfrom ,pos1
					      below (- (length ,shape) ,pos2)
					    do (progn
						 (when (and ,out-omit-p
							    (not (null (nth ,pos ,undetermined-shape-tmp)))
							    (not (shape-equal-1 ,pos ,undetermined-shape-tmp ,shape)))
						   ;; More details
						   ;; mismatch axes-originated.
						   (if (= (length ,undetermined-shape-tmp) (length ,shape))
						       (push
							(format
							 nil
							 "Couldn't idenfity ~~: ~~ is determined as ~a ~% butgot: ~a.~% Excepted ~~ = ~a, butgot: ~a"
							 (nth ,pos ,undetermined-shape-tmp) (nth ,pos ,shape)
							 (replace-nil ,undetermined-shape-tmp)
							 ,shape)
							,all-conditions)
						       ;; the mismatch of dimehsions originated error.
						       (push
							(format
							 nil
							 "The length of ~~ do not match.~%Excepted ~~ = ~a, butgot: ~a"
							 (replace-nil ,undetermined-shape-tmp)
							 ,shape)
							,all-conditions)))
						 (num-major-setf (nth ,pos ,undetermined-shape-tmp) (nth ,pos ,shape)))))))
		  
		  (flet ((merge-and-determine (shapes names)
			   (let ((out
				   (map
				    'list
				    #'(lambda (s name)

					  ;; Use-priorities:
					  ;; 1. [a b]
					  ;; 2. ~

					  (or
					   (when (or (not (symbolp s))
						     (find s ,undetermined-symbols))
					     s)
					   (and
					    (unless ,allow-symbol
					      (push
					       (format
						nil
						"Failed to determine this symbol: ~a" name)
					       ,all-conditions)
					      t)
					    '~)))
				    shapes
				    names)))
			     out)))
		    (let* (,@(loop with tmp = (gensym)
			           for let in let-binding
				   collect
				   `(,(car let) (let* ((,tmp (progn ,@(cdr let)))
						       (,tmp (if (listp ,(car let))
								 (flatten (list ,tmp))
								 ,tmp))
						       (,(car let) (if (listp ,tmp)
								       (flatten (list ,(car let)))
								       ,(car let))))
						  (if (if (listp ,tmp)
							  (cl-waffe2/vm.generic-tensor::shape-equal-list ,tmp ,(car let))
							  (shape-equal ,tmp ,(car let)))
						      ,tmp
						      (progn
							(push
							 (format
							  nil
							  "~a is declared as ~a, but determined as ~a"
							  ',(car let)
							  ,tmp
							  ,(car let))
							 ,all-conditions)
							,(car let)))))))
		      (let ((,~symbol (remove '~ ,undetermined-shape-tmp :test #'symbol-eq)))
			(declare (ignorable ,~symbol))
			
			(values
			 ;; (list out-shape1 out-shape...n)
			 (list ,@(map 'list #'(lambda (arg)
						`(flatten
						  (merge-and-determine
						   (list ,@arg)
						   ',arg)))
				      out-state))
			 (reverse ,all-conditions)
			 ,rank-error-p
			 (list ,@(map 'list #'(lambda (arg)
						`(flatten (list ,@arg)))
				      first-state))))))))))
      (when macroexpand
	(print body))

      (values (if return-body
		  body
		  (compile nil body))
	      (map 'list
		   #'(lambda (sym)
		       (position sym (the list input-names) :test #'symbol-eq))
		   output-names)
	      (map 'list
		   #'(lambda (s)
		       ;; The first rank is broadcastable?
		       (symbol-eq (car s) '~))
		   first-state)
	      (list first-state out-state)))))







;; ↓ Ituka yarou...

;; Refactor Version...
(defun expand-rank-check-forms (input-names first-states prev-subscripts detected-errors rank-err)
  "Comparing all ranks between first-state and prev-subscripts.
If any, appends Rank-Error to detected-errors"
  `(progn
     ;; Iterate by args (e.g.: A B C...)
     ,@(loop for name  in input-names
	     for state in first-states
	     for nth fixnum upfrom 0
	     collect `(when (= ,(length state) (nth ,nth ,prev-subscripts))
			(setq ,rank-err t)
			(push
			 (make-rank-error
			  :position      ,nth
			  :excepted-rank ,(length state)
			  :butgot        (length (nth ,nth ,prev-subscripts)))
			 ,detected-errors)))))

(defun expand-least-rank-check-forms (input-names first-states prev-subscripts detected-errors rank-err)
  `(progn
     ,@(loop for name in input-names
	     for state in first-states
	     for nth fixnum upfrom 0
	     collect
	     (let ((least-dim (- (length state) (count '~ state :test #'symbol-eq))))
	       `(when (< (length (nth ,nth ,prev-subscripts)) ,least-dim)
		  (setq ,rank-err t)
		  (push
		   (make-rank-atleast-error
		    :position ,nth
		    :first-state ',state
		    :butgot (length (nth ,nth ,prev-subscripts)))
		   ,detected-errors))))))

(defun expand-number-of-args-check-forms (previous-subscripts input-states out-state detected-errors)
  `(unless (= (length ,previous-subscripts) ,(length input-states))
     (push
      (make-number-of-args
       :first-state ',input-states
       :out-state ',out-state
       :previous-subscripts ,previous-subscripts)
      ,detected-errors)))

(defun expand-flex-dim-ident-form (nth-argument
				   pos1 pos2
				   symbol shape
				   out-omit-p detected-errors
				   ~
				   &aux (pos (gensym)))
  (declare (ignore symbol))
  ;; out-omit-p == '~
  `(loop for ,pos upfrom ,pos1 below (- (length ,shape) ,pos2)
	 do ,(when out-omit-p
	       `(when (and (not (null (nth ,pos ,~)))
			   (not (shape-equal-1 ,pos ,~ ,shape)))
		  (push
		   (make-flex-mismatch-error
		    :position ,nth-argument
		    :excepted (nth ,pos ,~)
		    :butgot   (nth ,pos ,shape))
		   ,detected-errors)))
	    (num-major-setf (nth ,pos ,~) (nth ,pos ,shape))))

(defun expand-dim-ident-form (nth-arg pos1 pos2
			      symbol shape detected-errors
			      &aux (pos (gensym)))
  `(cond
     ((< ,nth-arg ,pos1)
      (when (and (not (null ,symbol))
		 (or
		  ;; ('a 'a) or (10 10), not ('a 1) (1 'a)
		  (and (symbolp ,symbol)
		       (symbolp (nth ,nth-arg ,shape)))
		  (and (numberp ,symbol)
		       (numberp (nth ,nth-arg ,shape))))
		 (not
		  (equal ,symbol (nth ,nth-arg ,shape))))
	(push
	 (make-shape-mismatch-error
	  :position ,nth-arg
	  :excepted ,symbol
	  :butgot   (nth ,nth-arg ,shape))
	 ,detected-errors))
      (num-major-setq ,symbol (nth ,nth-arg ,shape)))
     ((> (- (length ,shape) ,nth-arg) ,pos1)
      (let ((,pos (- (1- (length ,shape)) (- ,pos2 ,nth-arg))))
	(when (and (not (null ,symbol))
		   (or
		    ;; ('a 'a) or (10 10), not ('a 1) (1 'a)
		    (and (symbolp ,symbol)
			 (symbolp (nth ,nth-arg ,shape)))
		    (and (numberp ,symbol)
			 (numberp (nth ,nth-arg ,shape))))
		   (not
		    (equal ,symbol (nth ,pos ,shape))))
	  (push
	   (make-shape-mismatch-error
	    :position ,nth-arg
	    :excepted ,symbol
	    :butgot   (nth ,nth-arg ,shape))
	   ,detected-errors))
	(num-major-setq ,symbol (nth ,pos ,shape))))))

;; Refactoring isn't done yet.
(defun create-subscript-p1 (subscripts
			    &key
			      (allow-symbol nil)
			      (macroexpand nil)
			      (fixed nil)
			      (return-body nil)
			      (local-variables nil)
			    &aux
			      (rank-error-p (gensym "rank-err-p"))
			      (previous-subscripts (gensym "PreviousShape"))
			      (known-symbols (gensym))
			      (detected-errors     (gensym "err")))
  "Returns a inlined version of subscript checker

Inputs:
    allow-symbol - If t, never produce error even when the predicted output includes symbols
    fixed        - If t, ~ is ignored.

Return: (values body output-names first-state (list first-state out-state))
"
  (declare (type list local-variables))
  (multiple-value-bind (input-names
			output-names
			first-state
			out-state
			let-binding)
      (parse-subscript subscripts :fixed fixed)
    (declare (optimize (speed 3))
	     (type list input-names output-names first-state out-state let-binding))

    ;; input-names  ... (A B)
    ;; output-names ... (C D)
    ;; first-state  ... ((~ x y z) (i j))
    ;; output-state ... ((~ x y z) (i j))

    (print input-names)
    (print output-names)
    (print first-state)
    (print out-state)
    (print let-binding)

    (let ((out-omit-p (find '~ (the list (flatten out-state)) :test #'symbol-eq))
	  (initial-symbols))

      ;; Register all symbols to be determined to initial-tables
      (mapc #'(lambda (name)
		(unless (find name initial-symbols :test #'symbol-eq)
		  (push name initial-symbols)))
	    (flatten first-state))

      (mapc #'(lambda (name)
		(unless (find name initial-symbols :test #'symbol-eq)
		  (push name initial-symbols)))
	    (flatten out-state))

      ;; ~ can be used at once and is a special symbol!
      ;; initial-symbols = (~ i j k ...)

      
      (let* ((~ (find '~ initial-symbols :test #'symbol-eq))
	     (flexdim-n (loop for input in first-state ;; the list of arguments which includes ~
			      for k fixnum upfrom 0
			      if (find '~ (the list (flatten input)) :test #'symbol-eq)
				collect k))
	     (body `(lambda (,previous-subscripts
			     &aux
			       (,known-symbols (find-symbols (flatten ,previous-subscripts)))
			       (,rank-error-p)
			       (,detected-errors))

		      ;; In the context, ~ must be used as the same rank
		      ,(when (and flexdim-n (not fixed))
			 `(let ((dim)
				(accd))
			    (loop for k in ',flexdim-n
				  if (null dim)
				    do (setq dim (length (nth k ,previous-subscripts)))
				       (setq accd k)
				  unless (= dim (length (nth k ,previous-subscripts)))
				    do (push
					(make-rank-mismatch-error
					 :position k
					 :excepted (length accd)
					 :butgot   (length (nth k ,previous-subscripts)))
					,detected-errors))))
		      
		      ;; Binding where a = 1 / a = (shape x) ...
		      ;; Place undetermined symbols (as long as they wasn't initialized by where)
		      (let* (,@let-binding
			     ,@(loop for symbol in initial-symbols
				     unless (or
					     (symbol-eq symbol '~)
					     (find symbol let-binding :key #'car :test #'symbol-eq))
				       collect (if (find (the symbol symbol) (the list local-variables))
						   `(,symbol ,symbol)
						   `(,symbol)))
			     ;; ~ = (nil nil nil) of max dim
			     ,@(when ~ ;; ~ was once used?
				 `((,~ (loop for s upfrom 0 below (loop for p in ,previous-subscripts maximize (length p))
					     collect nil)))))
			(declare (ignorable ,@initial-symbols))
			


			,(if fixed
			     ;; If ~ weren't used, do a rank check
			     ;; symbols could be also a list: A[after] where after = (shape ...)
			     ;; So do it after placing let-binding
			     
			     (expand-rank-check-forms
			      input-names
			      first-state
			      previous-subscripts
			      detected-errors
			      rank-error-p)

			     ;; even if ~ were used in subscript
			     ;; (1) for [~ i j] is still invaild.
			     (expand-least-rank-check-forms
			      input-names
			      first-state
			      previous-subscripts
			      detected-errors
			      rank-error-p))

			;; checking number of arguments
			;; ((1 2)) with A[i j] -> B[i j] is invaild for example.

			,(expand-number-of-args-check-forms
			  previous-subscripts
			  first-state
			  out-state
			  detected-errors)

			;; Starting Predicting Output Shapes
			;; Create and fullfil table			

			,@(loop for subscript in first-state
				for i fixnum upfrom 0
				collect
				;; Iterate by Arguments
				;; input = (~ i j) (~ j i) ...
				;; nth   = 0 1 2...

				;; ~ is flexible
				;; 0 1 ... ~
				;; ~ 0 1 ...
				`(progn
				   ,@(loop
				       with pos1 = (or (position '~ (the list subscript) :test #'symbol-eq) (length subscript))
				       with pos2 = (or (position '~ (reverse subscript) :test #'symbol-eq) 0)
				       with shape = `(nth ,i ,previous-subscripts)
				       for nth fixnum upfrom 0
				       for symbol in subscript

				       if (symbol-eq symbol '~)
					 collect (expand-flex-dim-ident-form
						  nth
						  pos1 pos2
						  symbol shape
						  out-omit-p detected-errors ~)
				       else
					 collect (expand-dim-ident-form
						  nth pos1 pos1 symbol shape detected-errors))))

			
			;; At last, we can inference out-state from the table

			;; Lambda Must Return:
			;; (value (list out-shape1 out-shape2 ...) all-errors rank-error-p

			(values
			 (flet ((merge-and-inf (shapes symbols)
				  (map 'list #'(lambda (s name position)
						 (or (when (or (not (symbolp s)) ;; S = 1 2 3...?
							       (find s ,known-symbols :test #'symbol-eq))
						       s)
						     ,(if allow-symbol ;; allow-symbols=t -> returns known symbol
							  ~
							  `(make-shape-mismatch-error
							    :position position
							    :excepted nil
							    :butgot   name))))
				       shapes symbols (range 0 (length symbols)))))
			   (list ,@(map 'list #'(lambda (state) `(flatten (merge-and-inf (list ,@state) ',state))) out-state)))
			 (reverse ,detected-errors)
			 ,rank-error-p
			 (list ,@(map 'list #'(lambda (arg)
						`(flatten (list ,@arg)))
				      first-state)))))))

	(when macroexpand
	  (print body))

	(values (if return-body
		    body
		    (compile nil body))
		(map 'list
		     #'(lambda (sym)
			 (position sym (the list input-names) :test #'symbol-eq))
		     output-names)
		(map 'list
		     #'(lambda (s)
			 ;; The first rank is broadcastable?
			 (symbol-eq (car s) '~))
		     first-state)
		(list first-state out-state))))))


