
(in-package :cl-waffe2)

;; network.lisp provides macros that used to simplify network definition.

;; TODO: DOCSTRINGS

(defmodel (Encapsulated-Node (self node-func)
	   :slots ((node-func :initarg :node-func))
	   :initargs (:node-func node-func)
	   :on-call-> ((self x)
		       (funcall (slot-value self 'node-func) x))))

(defmethod on-print-object ((model Encapsulated-Node) stream)
  (format stream "
    ~a
" (slot-value model 'node-func)))


;; TODO: More than one arguments
;; print-object encapsulated-node
;; (asnode #'!tanh)
;; (asnode #'!view `(10 10)
(defmacro asnode (function &rest arguments)
  "
## [macro] asnode

(call (asnode #'!tanh) x)
(call (asnode #'!add 2.0) x) ;; x += 2.0
"
  (if arguments
      `(Encapsulated-Node #'(lambda (x) (funcall ,function x ,@arguments)))
      `(Encapsulated-Node ,function)))

(defmacro call-> (input &rest nodes)
  "
## [macro] call->

an sequence of models

X
|
[]
|
...
[ASNODE !view] + [arg1 arg2] ...
|
[]
|

(call-> x
       (slot-value self 'linear1)
       (asnode #'!tanh)
       (slot-value self 'linear2)"

  (let ((nodes (reverse nodes)))
    (labels ((expand-call-form (rest-inputs)
	       (if rest-inputs
		   `(call ,(car rest-inputs)
			  ,(expand-call-form (cdr rest-inputs)))
		   input)))

      (expand-call-form nodes))))

(defmacro defsequence (name (&rest args) &rest nodes)
  "
## [macro] defsequence

(defsequence MLP (in-features)
    \"Docstring (optional)\"
    (LinearLayer in-features 512)
    (asnode #'!tanh)
    (LinearLayer 512 256)
    (asnode #'!tanh)
    (LinearLayer 256 10))
..."
  (let ((documentation (if (stringp (car nodes))
			   (car nodes)
			   ""))
	(nodes (if (stringp (car nodes))
		   (cdr nodes)
		   nodes))
	(names    (loop for i in nodes collect (gensym "Composite")))
	(keywords (loop for i in nodes collect (intern (symbol-name (gensym "KW")) "KEYWORD"))))

    `(defmodel (,name (self ,@args)
		:slots (,@(loop for name in names
				for kw   in keywords
				collect `(,name :initarg ,kw)))
		:initargs ,(let ((result))
			     (loop for kw in keywords
				   for node in nodes
				   do (push node result)
				      (push kw result))
			     result)
		:on-call-> ((self x)
			    (call-> x
				    ,@(loop for name in names
					    collect `(slot-value self ',name))))
		:documentation ,documentation))))

		    
