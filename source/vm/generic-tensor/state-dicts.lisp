
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; State Dicts:
;;  A general interface to store/restore trained model weights.
;;   - Features we provide should be limited to the mimimum.
;;   - Eazy to implement converter between cl-waffe2 and other frameworks. (since various formats are there...)
;;   - Also saves the state of optimizers (Adam?)

;;
;; - How to trace all parameters with their slot name?
;;  - 1. the method (call :around) finds all parameter from the slots of composite.
;;  - 2. In that time, the method also writes two slots of parameter tensors:
;;      - tensor-state-dict-name (symbol) tensor-param-belongs-to (composite)
;;      - represents the name of slot, the model belongs to respectively.
;;
;; That is, tensors to be saved, must belongs to any Composites, otherwise the function produces warning.
;;
;; State Dict Naming Convention:
;;   - Follows this rule:
;;     "{PREFIX}:{COMPOSITE_NAME}.{NTH?}.{SLOT_NAME}"
;;     where PREFIX = "param"
;;           COMPOSITE_NAME = the name of model the parameter belongs to (tensor-param-belongs-to ...)
;;           SLOT_NAME      = the name of slot  (tensor-state-dict-name ...)
;;           NTH            = As {COMPOSITE_NAME}.{SLOT_NAME} naming conflicts in the dictionary, NTH is increased by 1. First=0.
;;
;;     COMPOSITE_NAME, SLOT_NAME is a symbol but the package name is removed. (i.e.: cl-waffe2/nn:LinearLayer is saved as just linearlayer)
;;     All strings are downcased.
;;
;; Example: cl-waffe2/nn:LinearLayer has two parameter having these state_dict_name:
;;          "param:linearlayer.0.weight"
;;          "param:linearlayer.0.bias"
;; If AbstractOptimizer is hooked to the parameter, and it also have a parameter. (e.g.: Adam decay M, V, single-float)
;;           these status are saved in the following naming convention:
;;           optimizer:{STATE_DICT_NAME}.{OPTIMIZER_NAME}.{TYPE_SPECIFIER}.{SLOT_NAME}
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;; [TODO] Checkpoints, saving the status of optimizers.

;; Workloads
;;
;; - Optimizer Checkpoints
;; - Tests
;;  - APIs Docstrings, sections for it.
;;  - Exports
;;  - printings
;;  - .tf loader zisaku dekiruyuoni?

(in-package :cl-waffe2/vm.generic-tensor)

(defstruct (State-Dict
	    (:constructor from-state-dict (state-dict-hash-table &aux (compiled-composite nil) (weights nil) (optimizers nil)))
	    (:constructor make-state-dict (compiled-composite &key (weights T) (optimizers T) &aux (state-dict-hash-table nil))))
  "
## [struct] State-Dict

```lisp
;; Creating from loaded state-dict
(from-state-dict state-dict-hash-table)
```

```lisp
;; Creating from compiled composite
(make-state-dict compiled-composite &key (weights T) (optimizers T))
```

A table object obtained from tracing all parameters of Compiled-Composite.

To ensure reproducibility, state-dict collects following contents:

- If weights=T, AbstractTensor where requires-grad=T (the tensor must belong to any slots of Composite, otherwise cl-waffe2 can't determine the key name)

- If optimizers=T, reading all slots of `AbstractOptimizer`, values that satisfies `(numberp value)` `(typep x AbstractTensor)` are saved.

### State-dict naming convention

Basically Follows this rule:

`STATE_DICT_NAME = {PREFIX}:{COMPOSITE_NAME}.{NTH?}.{SLOT_NAME}`

where:

- `PREFIX` is one of param, missing_param, optimizer, missing_optimizer. if the prefix has `missing`, the parameter didn't belong to any composite. the prefix param indicates the corresponding value is a trained weight. optimizer indicates the value is one of slot of AbstractOptimizer.

- `COMPOSITE_NAME` = the name of model the parameter belongs to (tensor-param-belongs-to ...)

- `SLOT_NAME` = the name of slot  (tensor-state-dict-name ...)

- `NTH` As {COMPOSITE_NAME}.{SLOT_NAME} naming conflicts in the dictionary, NTH is increased by 1. First=0.

If the value to save is a slot of AbstractOptimizer, we use following naming in addition to STATE_DICT_NAME.

`optimizer:{STATE_DICT_NAME}.{OPTIMIZER_NAME}.{TYPE_SPECIFIER}.{SLOT_NAME}`

For example, LinearLayer has two parameters named as: `param:linearlayer.0.bias` `param:linearlayer.0.weights`. If adam optimizers are hooked to them, following keys are added: `optimizer:linearlayer.0.bias.adam.single-float.lr`, `optimizer:linearlayer.0.bias.adam.single-float.eps`, `optimizer:linearlayer.0.bias.adam.single-float.beta1`, `optimizer:linearlayer.0.bias.adam.single-float.beta2`, `optimizer:linearlayer.0.bias.adam.bit.n`, `optimizer:linearlayer.0.bias.adam.cputensor.m`, `optimizer:linearlayer.0.bias.adam.cputensor.v`, `param:linearlayer.0.weights`, `optimizer:linearlayer.0.weights.adam.single-float.lr` `optimizer:linearlayer.0.weights.adam.single-float.eps`, `optimizer:linearlayer.0.weights.adam.single-float.beta1`, `optimizer:linearlayer.0.weights.adam.single-float.beta2`, `optimizer:linearlayer.0.weights.adam.bit.n`, `optimizer:linearlayer.0.weights.adam.cputensor.m`, and `optimizer:linearlayer.0.weights.adam.cputensor.v`.

Note that all keys are stored as a string. all strings are downcased. The package to which the symbol belongs is ignored. (e.g.: cl-waffe2/nn:LinearLayer is saved as just linearlayer).

### Parsing a state dict key

In order to parse the state_dict key, the function `parse-state-dict-key` is available.

```lisp
(parse-state-dict-key key)
```

### Slots

`(state-dict-table state-dict)[hash-table]` key -> value hash table where :test is #'equal

"
  (table
   (or state-dict-hash-table
       (when compiled-composite
	 (make-state-dict-table compiled-composite :weights weights :optimizers optimizers))
       (error "make-state-dict: specify compiled-composite"))
   :type hash-table))

;; (defmethod print-object

(defun read-tinfo (tensor)
  "Reads state-dict-name and belongs-to"
  (assert (slot-value tensor 'requires-grad)
	  nil
	  "Assertion Failed: read-tinfo received a tensor which is not a parameter")
  
  (values (tensor-state-dict-name tensor) (tensor-param-belongs-to tensor)))

;; Ref: https://discuss.pytorch.org/t/what-numbering-convention-is-used-in-the-keys-of-a-state-dictionary/19758
(defun make-tensor-saved-name (attribute &rest args)
  ;; When counting up the duplicates: set nth=NIL.
  (format nil "~(~a~):~(~a~)"
	  attribute
	  (apply #'concatenate
		 'string
		 (butlast (loop for arg in args
				append
				(if (typep arg 'cl-waffe2/vm.nodes:Composite)
				    `(,(format nil "~(~a~)" (class-name (class-of arg))) ".")
				    `(,(format nil "~(~a~)" arg) ".")))))))


(declaim (ftype (function (Compiled-Composite &key (:weights boolean) (:optimizers boolean)) hash-table) make-state-dict-table))
(defun make-state-dict-table (compiled-composite &key (weights T) (optimizers NIL))
  "Once this function is called for compiled-composites, all params in the model become save/restore available."
  (let ((params (model-parameters compiled-composite))
	(naming-count (make-hash-table :test #'equal))
	(state-dict   (make-hash-table :test #'equal))
	(missing-info-list)
	(ok-list))

    ;; Updates tensor-state-dict-nth information given parameters.
    (dolist (tensor params)
      (multiple-value-bind (dict-name belongs-to) (read-tinfo tensor)
	(if (or (null dict-name)
		(null belongs-to))
	    (push tensor missing-info-list)
	    ;; Explores the count of confliction by setting nth=NIL.
	    (let ((key (make-tensor-saved-name "param" belongs-to nil dict-name)))
	      (push tensor ok-list)
	      (if (null (gethash key naming-count))
		  (setf (gethash key naming-count) 0)
		  (incf (gethash key naming-count)))
	      (setf (tensor-state-dict-nth tensor) (gethash key naming-count))))))

    (flet ((add-helper (tensor &key (prefix "param"))
	     (when weights
	       (setf
		(gethash
		 (make-tensor-saved-name
		  prefix
		  (tensor-param-belongs-to tensor)
		  (tensor-state-dict-nth   tensor)
		  (tensor-state-dict-name  tensor))
		 state-dict)
		tensor))
	     (when (and optimizers
			(tensor-optimizer tensor))
	       (let ((params (cl-waffe2/vm.nodes:find-params (tensor-optimizer tensor))))
		 (dolist (p params)
		   (setf
		    (gethash
		     (make-tensor-saved-name
		      (if (string= "param" prefix)
			  "optimizer"
			  "missing_optimizer")
		      (tensor-param-belongs-to tensor)
		      (tensor-state-dict-nth   tensor)
		      (tensor-state-dict-name  tensor)
		      (class-name (class-of (tensor-optimizer tensor)))
		      (type-of (cdr p))
		      (car p))
		     state-dict)
		    (cdr p)))))))

      (mapc #'add-helper ok-list)
      
      (when missing-info-list
	(warn "make-state-dict-table: Following tensors are created as requires-grad-p=T but cannot save as state_dict because they do not belongs to any slots of Composites.
When reading:
~a "
	      compiled-composite)
	(dolist (tensor missing-info-list)
	  (setf (tensor-param-belongs-to tensor) (gensym "?")
		(tensor-state-dict-nth   tensor) (gensym "NTH")
		(tensor-state-dict-name  tensor) (gensym "NAME"))
	  (warn "The parameter ~a cannot be saved as state_dict.
Temporary saved as: ~a
This parameter can't be restored when loading this table without reconfiguration of generated json file."
		tensor
		(make-tensor-saved-name
		 "missing_param"
		 (tensor-param-belongs-to tensor)
		 (tensor-state-dict-nth   tensor)
		 (tensor-state-dict-name  tensor)))
	  (add-helper tensor :prefix "missing_param")))
      state-dict)))

(trivia:defpattern state-dict-key (prefix-list content) `(trivia.ppcre:split ":" ,prefix-list ,content))
		   
		
(defun parse-state-dict-key (key)
  (declare (type string key))
  (trivia:ematch key
    ((state-dict-key (or "param" "missing_param" "optimizer" "missing_optimizer") content)
     (let ((prefix (string-upcase (cl-ppcre:regex-replace "_" (car (cl-ppcre:split ":" key)) "-"))))
       (apply #'values (intern prefix "KEYWORD")
	      (cl-ppcre:split "\\." content))))))

;; functions save-weights/load-weights are only defined for Compiled-Composite!
(defun save-weights ())
(defun load-weights ())
