
;;
;; lazy-subscript.lisp provides a solver for dynamic shaping.
;; Since cl-waffe2 tensor supports dynamic shaping and changed later, arithmetic operations used cl-waffe2 apis (e.g.: !reshape) 
;; have to deal with lazy symbols.
;; For Example, Considering flatten A(N 1 2 3), (apply #'!reshape a (* N 1 2 3)) should not work.
;; with ~ macro, can be written as: (apply #'!reshape (~ a b -> (* a b))).
;;
;; This file provides functional programming-like features dedicated to lazy-subscript.
;;

;; Subscript DSL is a high order function:
;;
;; DYNAMIC_SHAPE = {N=10, 20}
;;   Node Constructions:
;;     Conv2D where N C H W -> N C-out h-out w-out where C-out = ... H-out = ... W-out = ...
;;        ... = LazyAxis: f(DYNAMIC_SHAPE, MORE_SYMBOLS) -> FIXNUM
;;
;; =(x: LazyAxis, Y:T) := LazyAssertNode(observe(X) = Y)
;;
;; forward is a function where computes the next inputs of nodes/composites
;; (forward model A[10 10 10] B[10 10 10]) : All shapes are determined. Shape Transformation is instantly executed
;; (forward model A[10 A B]   B[10 A B]    : All shapes are NOT determined. shape computation is also lazily evaluated.

(in-package :cl-waffe2/vm)

;; Specs/Changes:
;;  Tensor.shape = (list LazyShape[0] LazyShape[1] LazyShape[2] ...)

;; Usage:
;;   (!reshape x (transform N C H W -> (* N C H) W))
;;   (%transform x[N C H W] -> [(* N C H) W]) (TODO)

;; Goals:
;;   (!reshape x (transform N C H W -> (* N C H) W)) ... Reshape/Permute/Viewの数はlambdaにすべき(Style)
;;   (call (conv2d 3 6 `(5 5)) 3 3)
;;


;; make-lazyaxis (Lazy and Encapsulate)
;;  -> (shape tensor) (tensor-view ...) and observe the result.


(eval-when (:compile-toplevel :load-toplevel :execute)

(defstruct (LazyAxis
	    (:constructor %make-lazyaxis (arguments form step)))
  (arguments arguments :type list)
  (form      (if (listp form)
		 (parse-lazy-exp form)
		 form))
  (constraints nil  :type list)
  (step        step :type function)
  (read-as     nil)
  (id          (gensym "axis")))

(defstruct (LazyAssertion
	    (:constructor make-lazy-assert (f evaluated-to)))
  "
## [struct] LazyAssertion
subject is f compared to evaluated-to
e.g.: A is = compared to 2
"
  (f f :type (member := :>= :<= :> :<))
  (evaluated-to evaluated-to))

(defmethod print-object ((lazyaxis LazyAxis) stream)
  (if (lazyaxis-arguments lazyaxis)
      (if (= (length (lazyaxis-arguments lazyaxis)) 1)
	  (format stream "LazyAxis: ~a"
		  (lazyaxis-form lazyaxis))	  
	  (format stream "LazyAxis: f~a = ~a"
		  (lazyaxis-arguments lazyaxis)
		  (lazyaxis-form lazyaxis)))
      (format stream "LazyAxis: ~a"
	      (lazyaxis-form lazyaxis))))

(defstruct (LazyIR
	    (:constructor make-lazyIR (type car cdr)))
  (type type :type (member :number :dynamic-shape :rest :arithmetic :function))
  (car  car)
  (cdr  cdr))

(defun ir->list (lazyir)
  (declare (type lazyir lazyir))
  (ecase (lazyir-type lazyir)
    (:number        (lazyir-car lazyir))
    (:dynamic-shape (lazyir-car lazyir))
    (:rest          (lazyir-car lazyir))
    (:arithmetic   `(,(lazyir-car lazyir) ,@(map 'list #'ir->list (lazyir-cdr lazyir))))
    (:function     `(,(lazyir-car lazyir) ,@(map 'list #'ir->list (lazyir-cdr lazyir))))))

(defun interpret-lazy (lazyir)
  (declare (type lazyir lazyir))
  (ecase (lazyir-type lazyir)
    (:number
     (let ((out (lazyir-car lazyir)))
       (if (numberp out)
	   out
	   (progn
	     (setf (lazyaxis-read-as out) nil)
	     (observe-axis out)))))
    (:dynamic-shape
     (let ((out (observe-variable (cadr (cadr (lazyir-car lazyir))))))
       (if (numberp out)
	   out
	   ;; Still out=symbol?
	   ;; Maybe the out is LazyAxis and undetermined due to dependencies
	   ;; Compute out first if out is LazyAxis.
	   (if (symbol-lazyaxis out)
	       ;; [MEMO] Detecting Circular Dependencies?
	       (progn
		 (observe-axis (symbol-lazyaxis out)))
	       (error "interpret-lazy: Encountered undeclared a dynamic shape: ~a" out)))))
    (:rest          (let ((out (second (lazyir-car lazyir))))
		      (setf (lazyaxis-read-as out) nil)
		      (observe-axis out)))
    (:arithmetic (apply (the symbol (lazyir-car lazyir)) (map 'list #'interpret-lazy (lazyir-cdr lazyir))))
    (:function   (apply (the symbol (lazyir-car lazyir)) (map 'list #'interpret-lazy (lazyir-cdr lazyir))))))

(defmethod print-object ((lazyir LazyIR) stream)
  (format stream "~a"
	  (ecase (lazyir-type lazyir)
	    (:number        (lazyir-car lazyir))
	    (:dynamic-shape
	     ;; (READ_SYMBOL 'A) -> 'A
	     (cadadr (lazyir-car lazyir))) 
	    (:rest
	     (format nil "{~a}" (second (lazyir-car lazyir))))
	    (:arithmetic
	     (with-output-to-string (out)
	       (format out "(")
	       (dotimes (nth (length (lazyir-cdr lazyir)))
		 (format out "~a" (nth nth (lazyir-cdr lazyir)))
		 (unless (= nth (1- (length (lazyir-cdr lazyir))))
		   (format out "~a" (lazyir-car lazyir))))
	       (format out ")")))
	    (:function
	     (with-output-to-string (out)
	       (format out "~(~a~)(" (lazyir-car lazyir))
	       (dotimes (nth (length (lazyir-cdr lazyir)))
		 (format out "~a" (nth nth (lazyir-cdr lazyir)))
		 (unless (= nth (1- (length (lazyir-cdr lazyir))))
		   (format out ", ")))
	       (format out ")"))))))

(defparameter *local-variable-table* nil)
(defun observe-variable (symbol)
  (or (and *local-variable-table*
	   (gethash symbol *local-variable-table*))
      (cl-waffe2/vm.generic-tensor::read-symbol symbol)))
			 
(defun parse-lazy-exp (exp)
  (trivia:ematch exp
    ((list* (or '+ '- '* '/) _)
     (parse-lazy-arithmetic (car exp) (cdr exp)))
    ((list* (or 'quote
		'#.(car ``nil))
	    _)
     (make-lazyir :number
		  exp nil))
    ((list* (type symbol) _)     
     (multiple-value-bind (ident) (cl-environments:function-information (car exp))
       (when (or
	      (eql ident :special-form)
	      (eql ident :macro))
	 (warn "parse-lazy-exp: Don't include a macro form as LazySubscript: ~a, otherwise produces the wrong result.

Describe the transformation of shapes as simple as possible.
   LazySubscript is consisted of these factors:
        - Functions
        - Numbers
        - Symbols
   Other factors that effects on the result, should be written at outside of :where."
	       exp)))
     (parse-lazy-function (car exp) (cdr exp)))
    ((type number)
     (make-lazyir :number
		  exp
		  nil))
    ((and (type symbol)
	  (not (type keyword)))
     (make-lazyir :dynamic-shape
		  `(observe-variable ',exp)
		  nil))
    ((type LazyAxis)
     (make-lazyir :rest
		  `(observe-axis ,exp)
		  nil))
    (_
     (make-lazyir :number
		  exp
		  nil))))

(defun parse-lazy-arithmetic (car cdr)
  (let ((args (map 'list #'parse-lazy-exp cdr)))
    (if (every #'(lambda (ir)
		   (and (eql (lazyir-type ir) :number)
			(typep (lazyir-car ir) 'fixnum)))
	       args)
	(make-lazyIR :number
		     (apply car (map 'list #'lazyir-car args))
		     nil)
	(make-lazyIR :arithmetic
		     car
		     args))))

(defun parse-lazy-function (car cdr)
  ;; [TODO]
  ;; Assert: car = function-name
  (let ((args (map 'list #'parse-lazy-exp cdr)))
    (if (every #'(lambda (ir)
		   (and (eql (lazyir-type ir) :number)
			(typep (lazyir-car ir) 'fixnum)))
	       args)
	(make-lazyIR :number
		     (apply car (map 'list #'lazyir-car args))
		     nil)
	(make-lazyIR :function
		     car
		     (map 'list #'parse-lazy-exp cdr)))))

(defun ir->args (ir)
  (let ((result))
    (labels ((helper (ir-of)
	       (when (eql (lazyir-type ir-of) :dynamic-shape)
		 (push (cadadr (lazyir-car ir-of)) result))
	       (dolist (var (reverse (lazyir-cdr ir-of)))
		 (helper var))))
      (helper ir)
      (delete-duplicates result))))

;;(defparameter *exp->compiled-cache* (make-hash-table :test #'equal))
(defun make-lazyaxis (expression)
  "
## [function] make-lazyaxis

```lisp
(make-lazyaxis expression)
```

Creates a LazyAxis from given expression.

```lisp
(make-lazyaxis `(* A B))   ;; LazyAxis: (A B) -> A * B
(make-lazyaxis `(floor A)) ;; LazyAxis: A     -> floor(A)
(make-lazyaxis 1)          ;; LazyAxis: 1
(make-lazyaxis `(+ 1 1))   ;; LazyAxis: 2
```

No macro usings are allowed; functions and fixnum, list are available.
"
  (let* ((lazyir (parse-lazy-exp expression))
	 (args   (ir->args lazyir))
	 (form   (ir->list lazyir)))

    (when (and
	   (typep (lazyir-car lazyir) 'fixnum)
	   (eql (lazyir-type lazyir) :number))
      (return-from make-lazyaxis (lazyir-car lazyir)))

    (when (eql (lazyir-type lazyir) :dynamic-shape)
      (return-from make-lazyaxis
	(cadr (second (lazyir-car lazyir)))))
    
    (%make-lazyaxis
     args
     lazyir
     (if (listp form)
	 #'(lambda () (interpret-lazy lazyir))
	 #'(lambda () form)))))

(defun make-dumpable-lazy-axis (expression)
  (let* ((lazyir (parse-lazy-exp expression))
	 (args   (ir->args lazyir))
	 (form   (ir->list lazyir)))
    (values
     args
     lazyir
     (if (listp form)
	 `(lambda () ,(ir->list lazyir))
	 `(lambda () ,form)))))

(defun make-higher-order-lazyaxis (arguments lazy-axis)
  "
## [function] make-higher-order-lazyaxis

(make-higher-order-lazy-axis
    `(A B 1)
    LazyAxis: f(A B C) = A*B*C)
=>
LazyAxis: f(A B) = A*B

Arguments can be received as:
    - observe-axis lazy-exp &rest args
    - dynamic shape table (global)
"
  (assert (= (length arguments) (length (lazyaxis-arguments lazy-axis)))
	  nil
	  "make-higher-order-lazyaxis: Assertion failed because the length of arguments do not match")

  (let ((remaining-args (loop for arg in arguments
			      for a   in (lazyaxis-arguments lazy-axis)
			      if (symbolp arg)
				collect arg)))
    (%make-lazyaxis
     remaining-args
     (parse-lazy-exp `(observe-axis ,lazy-axis ,@arguments))
     #'(lambda ()
	 (apply
	  #'observe-axis
	  lazy-axis
	  (loop for arg in arguments
		if (symbolp arg)
		  collect (observe-variable arg)
		else
		  collect arg))))))

;; Compose Several Axis (Arguments as LazyAxis)
(defun observe-axis (lazy-expression &rest args)
  "
## [function] observe-axis


```lisp
(observe-axis lazy-expression &rest args)
```
= funcall for lazy-expression

set the corresponding arg = nil to ignore.

Observes the result of LazyAxis obtained by calling lazyaxis-step function. The result is always fixnum.

LazyAxis: f(IN N) = floor((1+((IN+(2*N)+0+-1)/2)))
(observe-axis * 1 2)
"
  (declare (type LazyAxis lazy-expression))
  (or
   ;;(lazyaxis-read-as lazy-expression)
   (let ((vars (or *local-variable-table* (make-hash-table))))
     (loop for nth fixnum upfrom 0
	   for arg in (lazyaxis-arguments lazy-expression)
	   if (nth nth args)
	     do (setf (gethash arg vars) (nth nth args)))
     (let* ((*local-variable-table* vars)
	    (out (funcall (lazyaxis-step lazy-expression))))
       (if (or (listp out) (typep out 'fixnum))
	   (setf (lazyaxis-read-as lazy-expression) out)
	   out)))))

;; [TODO] Under this mode=T, changing adjustable shape is a invalid operation.
(defparameter *observe-mode* nil "
## [parameter] *observe-mode*

Set this parameter a list of tensors to determine all Lazy Subscripts when fixnum is needed (e.g.: when printing tensor, looping)

If this parameter is set to nil, maybe-observe-axis can return LazyAxis.")

(defmacro with-fixing-adjustable-shape ((&rest tensors) &body body)
  "under this macro, all adjustable shape is fixed"
  `(let ((*observe-mode* T))
     (dolist (tensor (list ,@tensors))
       (dolist (axis `(,@(slot-value tensor 'cl-waffe2/vm.generic-tensor::orig-shape)
		       ,@(slot-value tensor 'cl-waffe2/vm.generic-tensor::visible-shape)
		       ,@(slot-value tensor 'cl-waffe2/vm.generic-tensor::stride)
		       ,@(slot-value tensor 'cl-waffe2/vm.generic-tensor::input-shape)))
	 (when (typep axis 'LazyAxis)
	   (setf (lazyaxis-read-as axis) nil)
	   (observe-axis axis))))		      
     ,@body))

(defun tensor-fix-adjustable-shape (tensor)
  "Resets all results of adjustable shape belongs to tensor"
  (declare (type AbstractTensor tensor))
  (dolist (axis `(,@(slot-value tensor 'cl-waffe2/vm.generic-tensor::orig-shape)
		  ,@(slot-value tensor 'cl-waffe2/vm.generic-tensor::visible-shape)
		  ,@(slot-value tensor 'cl-waffe2/vm.generic-tensor::stride)
		  ,@(slot-value tensor 'cl-waffe2/vm.generic-tensor::input-shape)))
    (when (typep axis 'LazyAxis)
      (setf (lazyaxis-read-as axis) nil)
      (observe-axis axis)))
  nil)

(defun maybe-observe-axis (value)
  "Reads the given value as a fixnum.
value is expected as: LazyAxis, Symbol, Fixnum, rest...
If value is dynamic-shape -> observe it and returns as a fixnum
Otherwise                 -> Return as it is."
  (if (typep value 'LazyAxis)
      (if (every (compose #'numberp #'cl-waffe2/vm.generic-tensor::read-symbol) (lazyaxis-arguments value)) ;; Can determine?
	  (observe-axis value)
	  value)
      (if (symbolp value)
	  (if (symbol-lazyaxis value)
	      (observe-axis (symbol-lazyaxis value))
	      (cl-waffe2/vm.generic-tensor::read-adjustable-symbol value))
	  value)))

(defun maybe-observe-axis-no-err (value)
  (typecase value
    (LazyAxis
     (or (lazyaxis-read-as value)
	 (when *observe-mode*
	   (maybe-observe-axis value))
	 value))
    (symbol
     (cl-waffe2/vm.generic-tensor::read-symbol value))
    (number
     value)
    (T
     value)))

) ;; eval-when

;; [FIXME]
;; When I have enough time, Reimplement dynamic-shaping as much better ... elegant way.
(defparameter *lazyaxis->symbol* (make-hash-table :test #'equal))
(defparameter *symbol->lazyaxis* (make-hash-table :test #'equal))

(defmethod lazyaxis-symbol ((lazyaxis LazyAxis))
  (or (gethash (format nil "~a" lazyaxis) *lazyaxis->symbol*)
      (let ((id (gensym "LAZYAXIS")))
	(setf (gethash id *symbol->lazyaxis*) lazyaxis
	      (gethash (format nil "~a" lazyaxis) *lazyaxis->symbol*) id))))

(defun symbol-lazyaxis (symbol)
  (gethash symbol *symbol->lazyaxis*))


