
(in-package :cl-waffe2/base-impl)

;;

;; TODO
;; 1. using a list (%transform A[i j] -> [*to] where to = (shape A))
;; 2. [BugFix] (!view (randn `(3 3)) `(:broadcast 10) t) is OK??

;; Add: Reshape
;; Add: Test cases
;;

(defun parse-as-view-arg (arg)
  (typecase arg
    (list `(list ,@arg))
    (fixnum arg)
    (symbol
     (let ((symbol-str (symbol-name arg)))
       (if (equal #\* (aref symbol-str 0))
	   ;; *a is broadcast.
	   (let ((latter-part (subseq symbol-str 1 (length symbol-str))))
	     (if (typep (read-from-string latter-part) 'fixnum)
		 `(list :broadcast ,(read-from-string latter-part))
		 `(list :broadcast ,(intern latter-part))))
	   arg)))
    (T
     (error "%transform: unknown syntax of view: ~a" arg))))

;; TODO: This feature isn't enough.
(defmacro %transform (&body transform-syntax)
  "
## [macro] %transform

```lisp
(%transform &body transform-syntax)
```

`%transform` is a macro to describe `!view`, `!permute` and `broadcasting` of the given tensors together in a concise manner. In short word, `%transform = !view + !permute + Broadcasting`. The transformation of tensor are described on the same syntax of `Subscript DSL` but before and after `->`, there is always one tensor for each.

```
(Example)
(%transform A[i j] -> A[j i])
```

The variable names (e.g.: `A`) are exactly the name of the variable used by the `%transform` macro, which must be bound in scope. It is optional to give the name to the tensor after `->`.

```lisp
(defun transpose-revisit (tensor)
    (%transform tensor[~ i j] -> [j i]))
```

### Syntax

Following the rules below, `%transform` calls appropriate functions. If `~` were used after `->`, the macro is expanded into `!flexible ...`, or call `!permute` as long as all symbols appeared before `->` were also used after `->`. Otherwise, call `!view`.

### Adding an broadcastable axis.

The `broadcastable axis` is the range in which `1` of the shape of tensors can be added if needed, and at most one exists in one matrix.

If the subscripts of the tensor after `->` includes `~`, the corresponding position of the shape becomes `broadcastable`.

For example:

```lisp
(%transform A[i j] -> A[~ i j])
(%transform A[~ i j] -> A[~ i j])
```

### Adjustable dimensions

the `~` symbol used before `->` means: the number of dimensions of the corresponding part could be anything.

```lisp
(%transform A[~ i j] -> A[i j]
```

### Shuffling the permution of tensor

If symbols used before `->` are also appeared in after `->`, the corresponding symbols indicate the permution of tensor.

```lisp
(%transform A[i j] -> [j i])
(%transform A[~ i j] -> [j i])
(%transform A[i ~ j] -> [j i]) ;; the same as (!permute a 1 :~ 0)
```

### Make a view of tensors.

Set symbols (which aren't used before `->`) or fixnum to make a index. `(start end)` also creates a slice. Setting characters like `*10` `*a` broadcasts the axis.
"
  (multiple-value-bind (names-from names-to subs-from subs-to let-bindings) (parse-einsum-syntax `,transform-syntax)
    (declare (ignore names-to))

    (assert
     (= (length names-from)
	(length subs-from)
	(length subs-to)
	1)
     nil
     "%transform: Each subscripts appeared at once.
A[...] -> B[...]
^ Follow this form.")
    
    (let* ((subs-from (car subs-from))
	   (subs-to   (car subs-to))
	   (symbol-before (loop for s in (delete-duplicates (copy-list (alexandria:flatten subs-from))) if (not (symbol-eq s '~)) collect s))
	   
	   (~before (find '~ subs-from :test #'symbol-eq))
	   (~after  (find '~ subs-to   :test #'symbol-eq))
	   
	   (idx-table (make-symbol->idx-table symbol-before))
	   (add-broadcasting-pos (when ~after
				   (position '~ subs-to :test #'symbol-eq)))
	   
	   (permute-order (shape->ids-permute subs-to idx-table))
	   (read-variable (car names-from)))

      (when (and ~before ~after)
	(error "%transform: Invaild Syntax: ~a

the symbol ~~ can only appear either before or after `->`"
	       transform-syntax))
      
      (when (null read-variable)
	(error "%transform: The subscripts before ->, should be given its variable name."))

      ;; If ~before is T, ignore rank check
      `(let (,@let-bindings)
	 ,(when (not ~before)
	    `(when (not (= ,(length subs-from)
			   (dims ,read-variable)))
	       (error "%transform assertion: the rank do not match compared to the declared form")))

	 ;; Priorities = (Broadcast Permute View)
	 ,(cond
	    (~after
	     ;; ~ i j
	     ;; If the order is permuted
	     ;; Transform again
	     (let ((broadcast-out `(!flexible ,read-variable :at ,add-broadcasting-pos)))
	      ;; (print permute-order)
	       broadcast-out
	       ))
	    ((every #'numberp permute-order)
	     `(!permute ,read-variable
			,@(loop for before in subs-from
				if (symbol-eq before '~)
				  collect :~
				else
				  collect (pop permute-order))))
	    (T
	     ;; view
	     (let ((args (map 'list #'parse-as-view-arg subs-to)))
	       `(!view ,read-variable ,@args))))))))

(defun check-one-arg (form)
  (unless (= (length form) 1)
    (error "%transform: Each Subscript is applied to a single Tensor.
~a <- the length should be 1." form)))

(defmacro %transform-revisit (&body transform-syntax)
  "
## [macro] %transform

The macro `%transform` provides an abbreviated and much more readble syntax to compose these functions: `!flexible` `!view` `!reshape` `!permute`

```lisp
(%transform <<First State>> -> <<Second State>> -> <<Third State>> -> ... -> <<Last State>> where ...)
```

In `<<First State>>`, describe the shape of incoming tensor in this form:

```lisp
Tensor[Subscript]

A[i j]              ;; reading a variable a
A[~ i j]            ;; ~ -> the rank of the tensor is anything.
(randn `(3 3))[i j] ;; Setting function forms directly is ok.
```

Based on the first state of the tensor declared in `<<First State>>`, these operations are applied successively in subsequent forms.

### Permute

### View

### Adding an broadcastable axis

### Reshape

"
  (multiple-value-bind (var-names subscripts let-bindings)
      (cl-waffe2/vm.nodes::parse-transform-syntax `,transform-syntax)

    (let ((target-variable (car var-names))
	  (target-form     (car subscripts)))
      (check-one-arg target-variable)
      (check-one-arg target-form)
      (print var-names)
      (print subscripts)
      (print let-bindings)
      )))

