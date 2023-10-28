
(in-package :cl-waffe2/vm.nodes)

(defstruct Compiled-Subscript
  "
## [struct] Compiled-Subscript"
  (where nil :type list)
  (ignore-shape-error nil :type boolean)
  (compiled-f1 nil :type function)
  (compiled-f2 nil :type function))

(defun next-inputs (node compiled-subscript ignore-shape-error restart-case &rest inputs)
  (declare (optimize (speed 3))
           (type Compiled-Subscript compiled-subscript)
	   (type boolean ignore-shape-error)
	   (type (or null function) restart-case))

  (with-slots ((f1 compiled-f1) (f2 compiled-f2)) compiled-subscript
    (multiple-value-bind (out-state detected-errors) (funcall f1 inputs) ;; ... Finishes in < 1e-6 sec
      (if detected-errors
	  (progn
	    ;; If any errors occured, try again with removing ~ from subscripts. (I know this behaviour is ugly.)
	    (multiple-value-bind (out-state1 detected-errors-1) (funcall f2 inputs)
	      (if (and detected-errors-1
		       (not (or
			     ignore-shape-error
			     (compiled-subscript-ignore-shape-error compiled-subscript))))
		  (if restart-case
		      (funcall restart-case)
		      (describe-problems node detected-errors inputs out-state))
		  out-state1)))
	  out-state))))

(defun expand-lazy-let-binding (bind-to bind-what)
  "Creates a form that can receive symbols:
(expand-lazy-let-binding
   'a
   `(+ A B))
`(A (MAKE-HIGHER-ORDER-LAZYAXIS (LIST A B) LazyAxis: f(A B) = (A+B)))
"
  (multiple-value-bind (args form function)
      (cl-waffe2/vm::make-dumpable-lazy-axis bind-what)
    (declare (ignore form))
    ;; (observe-axis *) is enough to observe the result as long as
    ;; remaining symbols are provided as dynamic shape
    `(,bind-to (cl-waffe2/vm:make-higher-order-lazyaxis
		(list ,@args)
		(cl-waffe2/vm::%make-lazyaxis
		 '(,@args)
		 '(,@bind-what);; form can't dumped into fasl; in the first place this is not anymore needed print if needed.
		 ,function)))))

(defun a-and-b-maybe-equal (a b)
  (let ((a (if (listp a)
	       (flatten a)
	       (list a)))
	(b (if (listp b)
	       (flatten b)
	       (list b))))
    (cl-waffe2/vm.generic-tensor::shape-equal-list a b)))

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
	(expected-mode :variable-name)) ;; :symbol := :init-form
    (dolist (exp exps)
      (cond
	((and (symbol-eq expected-mode :variable-name)
	      (and (symbolp exp) ;; i.e.: exp is variable name
		   (not (symbol-eq exp '=))))
	 (setq var-name-tmp exp)
	 (setq expected-mode :=))
	((and (symbol-eq expected-mode :=)
	      (symbol-eq exp '=))
	 (setq expected-mode :init-form))
	((symbol-eq expected-mode :init-form)
	 (setq expected-mode :variable-name)
	 (push (list var-name-tmp exp) results))
	(T
	 (error 'subscripts-content-error
		:msg (format nil "Couldn't parse where-phase of subscripts.
Because the parser anticipated to come: ~a
But got: ~a.

Subscript: ~a
At       : ~ath Token, ~a"
			     expected-mode
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
		   :because :invalid-template-order
		   :target '->
		   :subscript subscripts
		   :msg "Please follow this template:
 Input-Shape -> Output-Shape where X = 0
(the where phase is optional.)
(Perhaps this is because of: The phase where never comes before -> phase comes.)"))

	  (when (and (not (null let-part))
		     (not (symbol-eq (car let-part) 'where)))
	    (error 'subscripts-format-error
		   :because :invalid-template-order
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

where [a[0~t]] -> [a[x]] let x = shape (local variables are accesible)

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

(defun build-subscript-note (nth-pos position called-as expected butgot states)
  (make-shape-error-message
   :position position
   :in-short   (format nil "~a should be ~a but got ~a." called-as expected butgot)
   :content    (format nil "the ~ath shape is ~a. ~a should be ~a but got ~a." nth-pos states called-as expected butgot)
   :suggestion (format nil "In ~a, set ~a=~a." states called-as expected)))


(defun find-symbols (list)
  (loop for l in list
	if (symbolp l)
	  collect l))

(defmacro num-major-setq (var obj)
  `(progn
     (when (typep ,var 'cl-waffe2/vm:LazyAxis)
       (push
	(cl-waffe2/vm:make-lazy-assert
	 :=
	 ,obj)
	(cl-waffe2/vm:lazyaxis-constraints ,var)))

     (when (typep ,obj 'cl-waffe2/vm:LazyAxis)
       (push
	(cl-waffe2/vm:make-lazy-assert
	 :=
	 ,var)
	(cl-waffe2/vm:lazyaxis-constraints ,obj)))

     (setq ,var
	 (or (when (or (numberp ,var)
		       (listp   ,var))
	       ,var)
	     ,obj))))

(defmacro num-major-setf (var obj)
  `(progn
     (when (typep ,var 'cl-waffe2/vm:LazyAxis)
       (push
	(cl-waffe2/vm:make-lazy-assert
	 :=
	 ,obj)
	(cl-waffe2/vm:lazyaxis-constraints ,var)))

     (when (typep ,obj 'cl-waffe2/vm:LazyAxis)
       (push
	(cl-waffe2/vm:make-lazy-assert
	 :=
	 ,var)
	(cl-waffe2/vm:lazyaxis-constraints ,obj)))
     (setf ,var
	   (or (when (or (numberp ,var)
			 (listp ,var))
		 ,var)
	       ,obj))))


(defun shape-equal-1 (pos list1 list2)
  (and (= (length list1)
	  (length list2))
       (shape-equal (nth pos list1) (nth pos list2))))

(defun replace-nil (list)  (map 'list #'(lambda (l) (if l l '?)) list))


;; TODO: Rewrite this ugly code, and print better error-notes
;; In order to use AOT Compiler, the function below is expands S-exp and seems ugly aaaa ><
;; I'm considering this refactoring
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
    ;; [a b c] [a b c] -> [a b c] let k = 1
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
	   ;; (1 2 3) for [x y] is invalid on the other hand (1 2 3) for [~ x y] is ok.
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
		
		;; If any, return error condition
		;; "Return: (values next-state condition)"
		;; TODO: Judge At-Least dims and return error.
		;; 1. Determine symbols, defined by let-binding
		

		(unless (= (length ,previous-subscripts)
			   (length ',least-required-dims))
		  (push
		   (make-number-of-args)
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
				  (make-shape-error-message
				   :position k
				   :in-short
				   (if (< dim (length (nth k ,previous-subscripts)))
				       "The Rank is too tall"
				       "The Rank is too low")
				   :content
				   (format nil "The given rank ~a do not match declared: ~a" dim (nth k ,previous-subscripts))
				   :suggestion
				   (if (< dim (length (nth k ,previous-subscripts)))
				       "Use (!rankup tensor -ntimes) to degrade the rank of tensors."
				       "Use (!flexible tensor) to explict a rank-up rule of broadcasting."))
				  ,all-conditions))))
			  
		;; The Number of dimensions error.
		(mapc
		 #'(lambda (nth-arg declared act)
		     (unless (<= declared (length act))
		       (and
			(setq ,rank-error-p t)
			(push
			 (make-shape-error-message
			  :position nth-arg
			  :in-short (format nil "The rank is incompatible")
			  :content  (format nil "is declared as: ~a and the rank must at least satisfy: >= ~a" (nth nth-arg ',first-state) declared)
			  :suggestion "If you believe this error is false, set (ignore-shape-error self) = t")
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
			 (make-shape-error-message
			  :position nth-arg
			  :in-short (format nil "Rank must be ~a" declared)
			  :content  (format nil "is declared as: ~a" (nth nth-arg ',first-state))
			  :suggestion "Check the definition.")
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
						   ;; ~ <- Broadcastingの範囲のShapeError
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
						      (build-subscript-note ,nth-arg ,i ',var ,var (nth ,nth-arg ,shape) ',subscript)
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
							(build-subscript-note ,nth-arg ,i ',var ,var (nth ,nth-arg ,shape) ',subscript)
							,all-conditions))
						     (num-major-setq ,var (nth ,pos ,shape)))))
				     else
				       collect ;; Batch-Size ...
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
							(make-shape-error-message
							 :position ,i
							 :in-short "Inconsistent use of batch: ~~."
							 :content  (format nil "~~ is already determined as ~a but the tensor attempted to make it ~a."
									   (nth ,pos ,undetermined-shape-tmp)
									   (nth ,pos ,shape))
							 :suggestion (format nil "the ~ath shape should be: ~a or broadcasted." (1+ ,pos) (nth ,pos ,undetermined-shape-tmp)))
							,all-conditions)
						       ;; the mismatch of dimehsions originated error.
						       (push
							(make-shape-error-message
							 :position ,i
							 :in-short "The length of ~~ do not match."
							 :content (format nil "the ~ath shape is ~a but it violates ~~ = ~a" (1+ ,pos) ,shape (replace-nil ,undetermined-shape-tmp))
							 :suggestion "")
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
					       (make-shape-error-message
						:position 0
						:place-me-last T
						:content (format nil "Failed to determine this symbol: ~a" name))
					       ,all-conditions)
					      t)
					    '~)))
				    shapes
				    names)))
			     out)))

		    ;; parsing where var = fixnum | var = list
		    (let* (,@(loop with tmp = (gensym)
			           for let in let-binding
				   collect
				   (let ((form (expand-lazy-let-binding tmp (second let))))
				     `(,(car let)
				       (let ((,tmp ,(second form)))
					 (if (= (length (cl-waffe2/vm:lazyaxis-arguments ,tmp)) 0)
					     (let ((,tmp (cl-waffe2/vm:observe-axis ,tmp)))
					       (if (a-and-b-maybe-equal
						    ,tmp
						    ,(car let))
						   ,tmp
						   (progn
						     (push
						      (make-shape-error-message
						       :place-me-last t
						       :content
						       (format nil
							       "~a should be ~a, but determined as: ~a"
							       ',(car let)
							       ,tmp
							       ,(car let)))
						      ,all-conditions)
						     ,tmp)))
					     (progn
					       ,tmp)))))))
		      
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
