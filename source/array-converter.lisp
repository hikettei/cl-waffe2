
(in-package :cl-waffe2)

;;
;; array-converter.lisp provides a converter between: WaffeTensor <-> Other Arrays
;;

;;
;; TODO: List, Array, Simple-Array, Sync
;;

(defgeneric convert-tensor-facet (from to)
  (:documentation "
## [generic] convert-tensor-facet

```lisp
(convert-tensor-facet from to)
```

The generic function `convert-tensor-facet` pays an important role when converting the data structure between `AbstractTensor` and other arrays (e.g.: `simple-array` etc...). Set `from` = `<<Array Before Converted>>`, and `to` = `(type-of <<Datatype you need>>)`, we dispatch the appropriate method and return converted arrays. Note that there's no assurance but before and after the converting, the pointers endeavour to indicate the same thing. If `AbstractTensor` to be converted has shuffled or viewed, we make a copy so that they become contiguous in memory.

This method is intended to be extended by users.

For example, converting `AbstractTensor` -> `simple-array`:

```lisp
(convert-tensor-facet (randn `(3 3)) 'simple-array)
```

### Performance Issues

In some conditions, `convert-tensor-facet` will run an additional compiling/copying which reduces poor performance. (FixME)

1. `AbstractTensor` to be converted is permuted, or is viewded.

```
CL-WAFFE2> (time (change-facet (permute* (ax+b `(4 3) 1 0) 0 1) :direction 'simple-array))
Evaluation took:
  0.009 seconds of real time
  0.008980 seconds of total run time (0.006915 user, 0.002065 system)
  100.00% CPU
  59 lambdas converted
  20,883,224 processor cycles
  2,220,176 bytes consed
  
#(0.0 3.0 6.0 9.0 1.0 4.0 7.0 10.0 2.0 5.0 8.0 11.0)
```

```lisp
(time (change-facet (view (ax+b `(4 3) 1 0) `(1 2)) :direction 'simple-array))
Evaluation took:
  0.010 seconds of real time
  0.010178 seconds of total run time (0.007312 user, 0.002866 system)
  100.00% CPU
  54 lambdas converted
  24,399,236 processor cycles
  1,958,160 bytes consed
  
#(3.0 4.0 5.0)
```

This because as of this writing we don't have any measures of moving tensors from non-contiguous to contiguous places, so reluctantly call `(proceed (->contigous tensor))`.

To avoid this, it is recommended to move the tensor into contigous place in advance, in the previous function.

See also: `convert-facet`

"))

(defun change-facet (array-from &key (direction 'array))
  "
## [function] change-facet

```lisp
(change-facet (array-from &key (direction 'array)))
```

Changes the facet of given `array-from` into `direction`. This function is just an alias for `convert-tensor-facet`

See also: `convert-tensor-facet`

### direction

As of this writing(2023/7/18), we provide these directions in default.

`array` returns ommon Lisp Array, with keeping the shape of tensors.

`simple-array` returns Common Lisp Array but 1D. the order of elements hinge on the order of `tensor.`

`AbstractTensor` returns `AbstractTensor` (devices to use depend on `*using-device*`). The dtype of returned tensor can be inferred from a first element of given array.

"
  (convert-tensor-facet array-from direction))


#|
[BugFix]: Stride Changes, Multidimensional Offsets are ignored.
;; When called with argument, we need to call permute* or view
;; Moves Place <- Target, and statically working.
(defmodel (Move-Into-Contiguous (self)
	   :where (Place[~] Target[~] -> Place[~])
	   :on-call-> ((self place target)
		       (declare (ignore self))
		       (!move place target :force t))))

(define-composite-function (Move-Into-contiguous) move-static)
|#


(defun AbstractCPUTensor->simple-array (from)
  ;; AbstractTensor -> Simple-Array
  (assert (cl-waffe2/vm.generic-tensor::vec from)
	  nil
	  "convert-tensor-facet: Assertion Failed because the given tensor doesn't have a existing vec.")

  ;; TODO: Check from is [computed]

  (cond
    ((cl-waffe2/vm.generic-tensor::permuted-p from)
     ;; Permuted?
     ;; FixME: Convert-Tensor-Facet with permuted is slow
     ;; because compiling is running.
     (tensor-vec (proceed (->contiguous from) :compile-mode :fastest)))
    ((tensor-projected-p from)
     ;; FIXME
     (tensor-vec (proceed (!copy from :force t) :compile-mode :fastest)))
    (T
     (tensor-vec from))))

;; AbstractTensor[CPU, Lisp] -> Simple-Array, Array
(defmethod convert-tensor-facet ((from CPUTensor)  (to (eql 'simple-array))) (AbstractCPUTensor->simple-array from))
(defmethod convert-tensor-facet ((from LispTensor) (to (eql 'simple-array))) (AbstractCPUTensor->simple-array from))


(defun simple-array->array (array dimensions dtype)
  (declare (type simple-array array))
  (make-array dimensions
	      :element-type (dtype->lisp-type dtype)
	      :displaced-to array
	      :displaced-index-offset 0))

(defmethod convert-tensor-facet ((from CPUTensor) (to (eql 'array)))
  (simple-array->array (AbstractCPUTensor->simple-array from)
		       (shape from)
		       (dtype from)))

(defmethod convert-tensor-facet ((from LispTensor) (to (eql 'array)))
  (simple-array->array (AbstractCPUTensor->simple-array from)
		       (shape from)
		       (dtype from)))


;; Simple-Array/Array/List -> AbstractTensor
(defmethod convert-tensor-facet ((from simple-array) (to (eql 'AbstractTensor)))
  (let ((storage-vec (progn
		       #+sbcl(sb-ext:array-storage-vector from)
		       #-sbcl(make-array (apply #'* (array-dimensions from))
					 :element-type (array-element-type from)
					 :displaced-to from))))
    (cl-waffe2/vm.generic-tensor::make-tensor-from-vec
     (array-dimensions from) (dtype-of (aref storage-vec 0)) ;; TODO: Fix it. If array-element-type=t, keep doing this, otherwise use it.
     storage-vec)))

(defmacro with-facet ((var (object-from &key (direction 'array) (sync nil))) &body body)
  "
## [macro] with-facet

```lisp
(with-facet (var (object-from &key (direction 'simple-array)) &body body))
```

The macro `with-facet` changes the facet of given `object-from` into `direction`, binding the result to `var`. If you want to apply modifications to `object-from` which applied inside `body`, set `sync`=`t`. (Only available when `object-from`=`AbstractTensor` otherwise ignored).

The macro `with-facet` is working on the flowchart below. Note that on some conditions, `(convert-tensor-facet)` will create an additional copy/compiling which may cause performance issue.

```lisp
[macro with-facet]
        ↓
[Set var <- (convert-tensor-facet object-from direction)] ⚠️ If tensor is viewed/permuted, an additional compiling is invoked!
        ↓
[Processing body]
        ↓
[If sync=t, (setf (tensor-vec object-from) (tensor-vec (convert-tensor-facet var 'AbstractTensor)))]
```

### Example

```lisp
(let ((a (randn `(3 3))))
    (with-facet (a* (a :direction 'simple-array))
        (print a*)
        (setf (aref a* 0) 10.0))
   a)

;; Operations called with simple-array a*, also effects on a.

#(0.92887694 -0.710253 1.2339028 -0.78008 1.6763965 0.93389416 -0.5691122
  1.6552123 -0.108502984) 
{CPUTENSOR[float] :shape (3 3)  
  ((10.0         -0.710253    1.2339028)
   (-0.78008     1.6763965    0.93389416)
   (-0.5691122   1.6552123    -0.108502984))
  :facet :exist
  :requires-grad NIL
  :backward NIL}
```

See also: `with-facets`
"
  `(let ((,var (convert-tensor-facet ,object-from ,direction)))
     (prog1
	 (progn ,@body)
       ,(when sync
	  `(progn
	     (when (not (typep ,object-from 'AbstractTensor))
	       (warn "with-facet: sync=t is ignored because object-from is not AbstractTensor"))

	     (when (typep ,object-from 'AbstractTensor)
	       (let ((tensor (convert-tensor-facet ,var 'AbstractTensor)))
		 (setf (tensor-vec ,object-from) (tensor-vec tensor)))))))))

(defmacro with-facets ((&rest input-forms) &body body)
  "
## [macro] with-facets

with-facet but input-forms are several.


```lisp
(with-facets ((a ((randn `(3 3)) :direction 'array))
              (b ((randn `(3 3)) :direction 'array)))
    (print a)
    (print b))
#2A((-0.020553567 -0.016298171 -2.0616999)
    (0.68268335 0.33567926 -0.79862773)
    (1.7132819 0.8081283 0.47327513)) 
#2A((-0.9344233 0.3149136 -0.8516832)
    (0.17137305 -0.026806794 -0.8192844)
    (0.19916026 -0.5102597 1.1834184)) 
```
"
  (labels ((expand-forms (rest-forms)
	     (if rest-forms
		 `(with-facet ,(car rest-forms)
		    ,(expand-forms (cdr rest-forms)))
		 `(progn ,@body))))
    (expand-forms input-forms)))


