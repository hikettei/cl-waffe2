
(in-package :cl-waffe2)

;;
;; array-converter.lisp: Here, we provide a common framework for the conversion between AbstractTensor and other matrix types.
;; The most basic method is convert-tensor-facet.
;; Other functions and macros work by assigning a method according to the type before and after the conversion and describing the process.
;; Conversions are performed while sharing pointers as much as possible.
;; If they cannot be shared, the with-facet macro forces a copy to be performed and pseudo-synchronises them.
;;


;; [TODO] Lists to AbstractTensor

(defgeneric convert-tensor-facet (from to)
  (:documentation "
## [generic] convert-tensor-facet

```lisp
(convert-tensor-facet from to)
```

Converts the given object (anything is ok; from=`AbstractTensor` `simple-array` etc as long as declared) into the direction indicated in `to`.

### Inputs

`From[Anything]` The object to be converted

`To[Symbol]` Indicates to where the object is converted

### Adding an extension

Welcome to define the addition of method by users. For example, `Fixnum -> AbstractTensor` convertion can be written like:

```lisp
(defmethod convert-tensor-facet ((from fixnum) (to (eql 'AbstractTensor)))
    (make-tensor from))

(print (change-facet 1 :direction 'AbstractTensor))

;;{SCALARTENSOR[float]   
;;    1.0
;;  :facet :exist
;;  :requires-grad NIL
;;  :backward NIL} 
```

If any object to AbstractTensor conversion is implemented, it is strongly recommended to add it to this method.

### Example

```lisp
(convert-tensor-facet (randn `(3 3)) 'simple-array)
```

See also: `convert-facet (more convenient API)`
"))

(defun change-facet (array-from &key (direction 'array))
  "
## [function] change-facet

```lisp
(change-facet (array-from &key (direction 'array)))
```

By calling the `conver-tensor-facet` method, this function can change the facet of given `array-form` into the `direction`. (Just an alias of `(convert-tensor-facet array-from direction)`)

See also: `convert-tensor-facet`

### Standard Directions

We provide these symbols as a `direction` in standard.

- `array`: Any Object -> Common Lisp Standard ND Array

- `simple-array`: Any Object -> Common Lisp Simple-Array

- `AbstractTensor`: Any Object -> AbstractTensor. If couldn't determine the dtype, dtype of the first element of `array-from` is used instead.
"
  (convert-tensor-facet array-from direction))

(defun AbstractCPUTensor->simple-array (from)
  (assert (cl-waffe2/vm.generic-tensor::vec from)
	  nil
	  "convert-tensor-facet: Attempted to convert the given AbstractTensor into Simple-Array but failed because the given tensor doesn't have an existing vec.
~a" from)

  ;; Make it contiguous on the memory?
  (tensor-vec from))

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

By using the convert-tensor-facet` method, this macro changes the facet of `object-from` into the `direction`. If you want to apply any operations to `object-from` and ensure that modifications are applied to the `object-from`, set `sync`=t and moves element forcibly (only available when direction=`'abstracttensor`). This is useful when editing AbstractTensor or calling other libraries without making copies.

For a more explict workflow, see below:

```lisp
    [macro with-facet]
            ↓
[Binding var = (convert-tensor-facet object-from direction)] 
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
	 (locally ,@body)
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

Bundles several `with-facet` macro.

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
		 `(locally ,@body))))
    (expand-forms input-forms)))

(defun ->tensor (object)
  "
## [function] ->tensor

Using the `convert-tensor-facet` method, converts the given object into AbstractTensor.

### Example

```lisp
(->tensor #2A((1 2 3) (4 5 6)))
```
"
  (convert-tensor-facet object 'AbstractTensor))

