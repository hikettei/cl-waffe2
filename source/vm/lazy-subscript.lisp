
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
;;

(in-package :cl-waffe2/vm)

;; Specs/Changes:
;;  Tensor.shape = (list LazyShape[0] LazyShape[1] LazyShape[2] ...)

;; How this behaviour appeared in the pragram?
;;   (!reshape x (transform N C H W -> (* N C H) W))
;;   (%transform x[N C H W] -> [(* N C H) W]) (TODO)

;; Dynamic Shape (with S-expression) System in cl-waffe2.

;; Basic Usage:
;;  1. AbstractTensor can include symbols as a shape, strides, views.
;;    - (N C H W) Tensor is ok for example and symbols are later changed.

;;  2. AbstractTensor is also needed to be include S-expression as a shape/stride/views
;;    - e.g.: when slicing (N C H W) Tensor, the result should be expressed in S-exp.
;;    - (I know this is ugly but) this file provides S-exp <-> Symbol hash-table
;;      - AbstractTensor interprets S-exp as a symbol because S-exp is replaced with randomly generated symbols in runtime.

;;  3. (make-lazyaxis S-exp) to create a lazy-S-exp
;;     (maybe-observe-axis symbol) to evaluate a LazyAxis.

;; In the node construction phase, all shapes are not necessary to be determined;
;; But we can determine them by comparing inputs given by (forward model ...) method.
;; Tensor Shapes and Subscript DSL is expressed in a high order lambda function.
;; Determines symbols step-by-step.

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

;; [FixME] This feature is not available for a now.
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
  (format stream "[~a]" (lazyaxis-form lazyaxis)))

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
  (declare (type LazyIR lazyir))
  (ecase (lazyir-type lazyir)
    (:number
     (let ((out (lazyir-car lazyir)))
       (if (typep out 'AbstractTensor)
	   (if (or
		(null (tensor-state out))
		(and
		 (tensor-state out)
		 (eql :computed (wf/t::state-name out (tensor-state out)))))
	       out
	       (or
		(when *static-alloc-state*
		  (gethash (tensor-id out) (vmalloc-id2pool *static-alloc-state*)))
		(error "interpret-lazy: ~a is not yet determined." out)))
	   (if (or (numberp out)
		   (eql out t)
		   (null out))
	       out
	       (progn
		 (setf (lazyaxis-read-as out) nil)
		 (observe-axis out))))))
    (:dynamic-shape
     (let ((out (observe-variable (cadr (cadr (lazyir-car lazyir))))))
       (if (numberp out)
	   out
	   ;; Still out=symbol?
	   ;; Maybe the out is LazyAxis and undetermined due to dependencies
	   ;; Compute out first if out is LazyAxis.
	   (if (symbol-lazyaxis out)
	       ;; [MEMO] Detecting Circular Dependencies?
	       (observe-axis (symbol-lazyaxis out))
	       (let ((result (wf/t::read-symbol out)))
		 (if (numberp result)
		     result
		     (error "interpret-lazy: Encountered undeclared a dynamic shape: ~a" out)))))))
    (:rest          (let ((out (second (lazyir-car lazyir))))
		      (setf (lazyaxis-read-as out) nil)
		      (observe-axis out)))
    (:arithmetic (apply (the symbol (lazyir-car lazyir)) (map 'list #'interpret-lazy (lazyir-cdr lazyir))))
    (:function
     (apply (the symbol (lazyir-car lazyir)) (map 'list #'interpret-lazy (lazyir-cdr lazyir))))))

(defmethod print-object ((lazyir LazyIR) stream)
  (format stream "~a"
	  (ecase (lazyir-type lazyir)
	    (:number
	     (if (typep (lazyir-car lazyir) 'AbstractTensor)
		 (format nil "~a{size=~a}" (tensor-id (lazyir-car lazyir)) (shape (lazyir-car lazyir)))
		 (lazyir-car lazyir)))
	    (:dynamic-shape
	     ;; (READ_SYMBOL 'A) -> 'A
	     (let ((sym (cadadr (lazyir-car lazyir))))
	       (or
		(symbol-lazyaxis sym)
		sym)))
	    (:rest
	     (format nil "{~a}" (second (lazyir-car lazyir))))
	    (:arithmetic
	     (with-output-to-string (out)
	       (format out "")
	       (dotimes (nth (length (lazyir-cdr lazyir)))
		 (format out "~a" (nth nth (lazyir-cdr lazyir)))
		 (unless (= nth (1- (length (lazyir-cdr lazyir))))
		   (format out "~a" (lazyir-car lazyir))))
	       (format out "")))
	    (:function
	     (if (eql (lazyir-car lazyir) 'wf/t::vref)
		 (format nil "~a[~a]" (car (lazyir-cdr lazyir)) (second (lazyir-cdr lazyir)))
		 (with-output-to-string (out)
		   (format out "~(~a~)(" (lazyir-car lazyir))
		   (dotimes (nth (length (lazyir-cdr lazyir)))
		     (format out "~a" (nth nth (lazyir-cdr lazyir)))
		     (unless (= nth (1- (length (lazyir-cdr lazyir))))
		       (format out ", ")))
		   (format out ")")))))))

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
	;; A+0 -> A Mutation
	(let ((args
		(if (eql car '+)
		    (loop for arg in args
			  if (or
			      (not (eql (lazyir-type arg) :number))
			      (not (eql (lazyir-car arg) 0)))
			    collect arg)
		    args)))
	  (if (and (eql car '+)
		   (= (length args) 1)
		   (numberp (car args)))
	      (make-lazyir :number
			   (car args)			       
			   nil)
	      (make-lazyIR :arithmetic
			   car
			   args))))))

(defun parse-lazy-function (car cdr)
  ;; [TODO]
  ;; Assert: car = function-name
  (let ((args (map 'list #'parse-lazy-exp cdr)))
    (if (every #'(lambda (ir)
		   (and (eql (lazyir-type ir) :number)
			(or (typep (lazyir-car ir) 'fixnum)
			    (typep (lazyir-car ir) 'boolean))))
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

    (when (typep (lazyir-car lazyir) 'boolean)
      (return-from make-lazyaxis (lazyir-car lazyir)))

    (when (eql (lazyir-type lazyir) :dynamic-shape)
      (return-from make-lazyaxis
	(cadr (second (lazyir-car lazyir)))))

    (when (and
	   (eql (lazyir-car lazyir) 'vref)
	   (typep (second (lazyir-cdr lazyir)) 'LazyIR)
	   (numberp (lazyir-car (second (lazyir-cdr lazyir))))
	   (let ((tensor (lazyir-car (car (lazyir-cdr lazyir))))
		 (val    (lazyir-car (second (lazyir-cdr lazyir)))))
	     (and
	      (typep tensor 'AbstractTensor)
	      (wf/t::vec tensor)
	      (or
	       (null (tensor-state tensor))
	       (eql :computed (wf/t::state-name tensor (tensor-state tensor))))
	      (return-from make-lazyaxis (floor (wf/t:vref tensor val)))))))
    
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

;;(declaim (inline maybe-observe-axis))
(defun maybe-observe-axis (value)
  "Reads the given value as a fixnum.
value is expected as: LazyAxis, Symbol, Fixnum, rest...
If value is dynamic-shape -> observe it and returns as a fixnum
Otherwise                 -> Return as it is."
  (if (typep value 'LazyAxis)
      ;; Can be determined?
      (if (let ((shape
		  (map 'list (compose #'numberp #'cl-waffe2/vm.generic-tensor::read-symbol) (lazyaxis-arguments value))))
	    (and
	     (every #'numberp shape)
	     (not (some #'(lambda (x) (= -1 x)) shape))))
	  (observe-axis value)
	  value)
      (if (symbolp value)
	  (if (symbol-lazyaxis value)
	      (if (= -1 (cl-waffe2/vm.generic-tensor::read-adjustable-symbol value))
		  (block try-make-it-static
		    (handler-bind
			((error #'(lambda (cond) (declare (ignore cond)) (return-from try-make-it-static -1))))
		      (observe-axis (symbol-lazyaxis value))))
		  (observe-axis (symbol-lazyaxis value)))
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

(defparameter *lazyaxis->symbol* (make-hash-table :test #'equal) "[A*B](*PACKAGE*) -> LAZYAXIS_N")
(defparameter *symbol->lazyaxis* (make-hash-table :test #'equal) "LAZYAXIS_N -> [A*B](*PACKAGE*)")

(defmethod lazyaxis-symbol ((lazyaxis LazyAxis))
  (or (gethash (format nil "~a~a" (package-name *package*) lazyaxis) *lazyaxis->symbol*)
      (let ((id (gensym "LAZYAXIS")))
	(setf (gethash id *symbol->lazyaxis*) lazyaxis
	      (gethash (format nil "~a~a" (package-name *package*) lazyaxis) *lazyaxis->symbol*) id))))

(defmethod lazyaxis-symbol ((lazyaxis T)) nil)

(defun symbol-lazyaxis (symbol)
  (gethash symbol *symbol->lazyaxis*))


(defparameter *lazy-asserts* nil)
(defun lazy-assert (A B &key (test '=))
  (when *lazy-asserts*
    (when (not (equal A B))
      (push (make-lazyaxis `(,test ,A ,B)) *lazy-asserts*))))

(defun node-realize-assertions (op)
  (when (typep op 'AbstractNode)
    (let ((vals (cl-waffe2/vm.nodes::node-lazy-asserts op)))
      (dolist (val vals)
	(when (not (eql val t))
	  (assert (wf/vm::maybe-observe-axis val) () "LazyAssertion Failed: ~a. ~%The shapes does not match." val))))))


