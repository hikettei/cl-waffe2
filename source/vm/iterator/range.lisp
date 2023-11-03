
(in-package :cl-waffe2/vm.iterator)

;; Examples: view と ~ の使い方追加する？

;;  !view
;;  !make-input -> !view
;;  !view but step = -1
;;  Broadcasting usages

;; Range is designed to replace view.

;; Range = (LIST FROM TO STEP <<LazyAxis>>)

;; LazyAxisの表示を単純化する

;; 追加する:
;;  do-ranges do-inlined-ranges
;;  Broadcasting:
;;  ( 1 ~ 1 )
;;      ^ ここの~は強制挿入
;;  ( ~ 1 1 )
;;    ^ 先頭/終端はOptional (RankError is needed)

;; 次やること: call-with-view / do-compiled-loopのRefactoring

;;これ解ける？
;;(~ ~ M N)
;;(~ M 1~ N)
;; (range 3 0)みたいなのを禁止する？

;; -1 がBLAS XXX de ikeruka
;; Broadcasting semantic update

;; !viewでflexible insert kanou ni suru

;; [TODO1]
;;  - package nicknames to all packs (OK)
;;  - make-tensor, make-input: ~ as a !flexible
;;     - 1~ 1を強制的に挿入する, shapeの構文
;;  - Lazy...複数の引数
;;  - Dynamic-Shape 手動で帰れるようにする (OK?)
;;  - LoopNode 実装 Dynamic-Shape更新するようにする (Unnecessary?)
;;  - Examples/MLPが動いているか？ -> Merge
;;  - Iterator BLAS系の命令でも -1 Direction OK?

;; 手動で再実装しない？
;; render.lisp
;; mref
;; call-with-view
;; do-compiled-loop

;; Range ... Dynamic Shape, LazyAxis対応したい (OK)

;; Ranges should satisfy following properties:

;; - (Range 1 9) |
;; - (Range 1 9 3) | これ合成してなんでも表現できるようにする？

;; Lazy-Index-Components:
;;  lazy ... 複数の引数呼び出せればおk
;;  Creates a tensor of Shape
;;  (3 3)
;;  ((0 1 2)
;;   (2 3 4))

;; Range ... Index/Symbolが扱える
;; Reverse/Slice/Broadcastingが表現できる
;; Iterationが生成できる
;; Tを使える
;; FROM/TOのT = 行列の最後のサイズ
;; STEPのT    = broadcasting but 相手の方に合わせる
;; Broadcastingの状態= Tensor側に付与する (view.lisp)
;; Rangeはここでは触れない

;; Lazy-LoopとDynamic-Shapeの任意編集を組み合わせてDiagnoal実装可能?
;; view.lisp <-> iterator.lisp通信 (Parsing etc...)

;; [TODO] Readmeかどっかに書く nicknames
;; cl-waffe2/vm.generic-tensor -> wf/t
;; cl-waffe2/vm.nodes          -> wf/nodes
;; cl-waffe2/vm.iterator       -> wf/iter
;; cl-waffe2                   -> wf/utils
;; cl-waffe2/distributions     -> wf/d
;; cl-waffe2/nn                -> wf/nn
;; cl-waffe2/vm                -> wf/vm
;; cl-waffe2/optimizers        -> wf/opt
;; cl-waffe2/base-impl         -> wf

;; 次やること
;; .range
;;  tensor-view equal 比較できてるの？
;; .do-compiled-loopを実装

;; Parse-View/Call-With-Viewとの統合

;; diagnoal:
;; (A=B (lazy-index-components (range 0 10)) (range 0 10))

(defun lazy-max (a b)
  "Given the condition that dynamic shape is given as positive fixnum, this function lazily computes maximum value of a and b."
  (if (eql a 0)
      b
      (if (eql b 0)
	  a
	  `(max ,a ,b))))

(defun lazy-min (a b)
  "Given the condition that dynamic shape is given as positive fixnum, this function lazily computes minimum value of a and b."
  (if (eql a 0)
      a
      (if (eql b 0)
	  b
	  `(min ,a ,b))))

(deftype Index-Symbol-T ()
  `(or fixnum symbol))

(defstruct (Range
	    (:constructor make-range (from to &optional (step 1))))
  (from from :type index-symbol-t)
  (to   to   :type index-symbol-t)
  (step step :type index-symbol-t))

(defun symbol-pprint-helper (value)
  "Displays the value as pretty as possible"
  (if (typep value 'symbol)
      (or
       (symbol-lazyaxis value)
       value)
      value))

(defun range-start-index (range)
  "Returns a number of starting point"
  (declare (type range range))
  (make-lazyaxis (lazy-min (range-from range) (range-to range))))

(defun range-size (range)
  "Computes the number of iterations of range as simple as possible:
Basically can be computed in this formula:
    floor(ABS(from - to) // STEP)"
  (declare (type range range))
  (if (and (numberp (range-step range))
	   (= (range-step range) 1))
      (if (and (numberp (range-from range))
	       (= 0     (range-from range)))
	  (make-lazyaxis (range-to range))
	  (make-lazyaxis
	   `(abs (- ,(range-from range)
		    ,(range-to   range)))))
      (make-lazyaxis
       `(floor
	 (abs (- ,(range-from range)
		 ,(range-to   range)))
	 (abs ,(range-step range))))))

(defun pprint-range-size (range)
  (declare (type range range))
  (make-lazyaxis
   `(floor
     (abs (- ,(symbol-pprint-helper (range-from range))
	     ,(symbol-pprint-helper (range-to   range))))
     (abs ,(symbol-pprint-helper (range-step range))))))

(declaim (ftype (function (Range fixnum) fixnum) range-nth))
(defun range-nth (range count)
  (declare (type range range)
	   (type fixnum count)
	   (optimize (speed 3)))
  ;; [TODO] If count < 0 -> count + range_size
  (let* ((a (the fixnum (maybe-observe-axis (range-from range))))
	 (b (the fixnum (maybe-observe-axis (range-to   range))))
	 (c (the fixnum (maybe-observe-axis (range-step range))))

	 ;; 10 ... 1 step=step <=> 1 ... 10, step=-step
	 (c (if (> a b)
		(let ((tmp a))
		  (setq a b
			b tmp)
		  (- c))
		c)))
    (declare (type fixnum a b c))
    (if (< c 0)
	(let ((tmp b))
	  (setq b a
		a (+ c tmp))))
    (the fixnum (+ a (the fixnum (* c count))))))

(define-compiler-macro range-nth (range count)
  `(let* ((a (the fixnum (maybe-observe-axis (range-from ,range))))
	  (b (the fixnum (maybe-observe-axis (range-to   ,range))))
	  (c (the fixnum (maybe-observe-axis (range-step ,range))))

	  ;; 10 ... 1 step=step <=> 1 ... 10, step=-step
	  (c (if (> a b)
		 (let ((tmp a))
		   (setq a b
			 b tmp)
		   (- c))
		 c)))
     (declare (type fixnum a b c))
     (if (< c 0)
	 (let ((tmp b))
	   (setq b a
		 a (+ c tmp))))
     (the fixnum (+ a (the fixnum (* c ,count))))))

(defmacro do-range ((var range) &body body)
  "Creates an iteration following the instruction of range.
Range can include dynamic shape.

Note that iterations generated by do-range is NOT A THREAD_SAFE!!"
  (alexandria:with-gensyms (count)
    `(dotimes (,count (the (unsigned-byte 64) (maybe-observe-axis (range-size ,range))))
       (let ((,var (range-nth ,range ,count)))
	 (declare (type (unsigned-byte 64) ,var))
	 ,@body))))

(defun range (from &optional (to `(1+ ,from)) (step 1))
  "
## [function] range

```lisp
(range from &optional (to `(1+ ,from)) (step 1))
```

Creates a range: `[from, to) where step=step`. This structure is dedicated to a do-range macro which generates an iteration following rules:

- starting from (min from, to)
- ends with (max from, to)
- if step < 0, the order is reversed.
- This macro asserts that `var` is in the range of [from, to).
"
  (flet ((parse (value)
	   (let ((out (make-lazyaxis value)))
	     (if (typep out 'LazyAxis)
		 (lazyaxis-symbol out)
		 out))))
    (let ((from (parse from))
	  (to   (parse to))
	  (step (parse step)))
      (assert (or
	       (not (numberp step))
	       (not (= step 0)))
	      ()
	      "range: do not create a range whose step is 0 otherwise loop continues forever.")
      (make-range from to step))))

(defmethod print-object ((obj Range) stream)
  (format stream
	  (if (numberp (pprint-range-size obj))
	      "<Range For ~a, [~a, ~a), step=~a>"
	      "<Range~%    For [~a]~%      from ~a~%      to ~a~%      stepby ~a>")	      
	  (pprint-range-size obj)
	  (symbol-pprint-helper (range-from obj))
	  (symbol-pprint-helper (range-to   obj))
	  (symbol-pprint-helper (range-step obj))))

(defun ifelse (condition then else) (if condition then else))
(defun lazy-ifelse (condition then else)
  (let ((condition (make-lazyaxis condition)))
    (if (eql condition T)
	then
	(if (eql condition nil)
	    else
	    `(ifelse ,condition ,then ,else)))))

(defun .range (range2 &optional (range1 nil))
  "
## [function] .range

```lisp
(range+ range2 &optional (range1 nil))
```

Interprets the value of range2 from a viewpoint of range1.

`Original Tensor -> range2 -> [View] -> range1`

```
Applying a further slicing:
    (Range 2 10 2) ;; range1
 +) (Range 0 4  2) ;; range2
 ------------------
    (Range 2 6 2)

is defined as:
 A = min(range1.from, range1.to)
 B = min(range2.from, range2.to)
 C = max(range2.from, range2.to)

new_range_from  = A + B
new_range_to    = A + C
new_range_step  = lcm(range1.step, range2.step)

i.e.:
{a_2x+b_2}∪{a_1x+b_1} 
```
"
  (declare (type Range range2)
	   (type (or null Range) range1))

  (when (null range1)
    ;; NIL(RANGE1(...))
    (return-from .range range2))

  ;; range1 = base
  ;; range2 = new

  ;; Dynamic Shape is asserted to be a positive number
  (let* ((upfrom1 (range-from range1))
	 (below1  (range-to   range1))

	 (upfrom2 (range-from range2))
	 (below2  (range-to   range2))

	 (step1   (range-step range1))
	 (step2   (range-step range2))

	 (offset (lazy-ifelse
		  `(> ,step1 0)
		  (lazy-min upfrom1 below1)
		  (lazy-max upfrom1 below1)))

	 (from  `(+ ,offset (* (signum ,step1) ,upfrom2)))
	 (to    `(+ ,offset (* (signum ,step1) ,below2)))
	 
	 (from1 (lazy-min from to))
	 (to1   (lazy-max from to))
	 (step  `(* (signum ,step2) (lcm ,step1 ,step2))))
    ;; [TODO]
    ;; - Adding an assertion
    ;; - Adding a lazy assertion
    (let ((result (range from1 to1 step)))
      (assert
       (or
	(not (numberp (range-size result)))
	(not (numberp (range-size range1)))
	(<= (range-size result) (range-size range1)))
       ()
       ".range: before and after the composition, the size of range ~a must not exceed ~a.
from ~a to ~a => ~a"
       (range-size result)
       (range-size range1)
       range1
       range2
       result)
      
      ;; [FixME] The assertion above is not enough...
      ;; e.g.: (.range (range 2 8 1) (range 2 8 1)) should produce an error
      ;; but interpreted as a vaild operation.
      result)))

