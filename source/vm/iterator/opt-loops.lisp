
(in-package :cl-waffe2/vm.iterator)

;; TODO: Producing a list of subscript, iterations from DSL like:
;; In the future release, convnode, im2col is removed since the parallelization
;; is depends on it

;;
;; for (i=0..1)
;;     for (j=0..2)
;;         x[i,j] = y[j, i]
;;

;; TO ADD: simdpack, unpack

;; TODO: Producing a Lisp-Like DSL
;;        -> GCC, Metal, CUDA, Lisp etc...

(defun generate-indices (n-rank)
  (loop for n upfrom 0 below n-rank
	collect
	(intern (format nil "gid~a" n))))

(defun trace-invocation (op source-tensor target-tensor &key (kernel-rank 1) (collapse t))
  "call-with-view -> invocation
Assertion: Shapes are already determined."
  (declare (ignore collapse))
  (assert (= kernel-rank 1) () "kernel-rank must be 1")
  (assert (= (dims source-tensor) (dims target-tensor)) () "Assertion failed with ~a and ~a should have the same ranks" source-tensor target-tensor)
  
  (make-action
   :op op
   :source (make-indexspace
	    source-tensor
	    :subscripts (generate-indices (dims source-tensor))
	    :sizes (wf/t::translate-adjustable-shape (shape source-tensor)))
   :target (make-indexspace
	    target-tensor
	    :subscripts (generate-indices (dims target-tensor))
	    :sizes (wf/t::translate-adjustable-shape (shape target-tensor)))))

(defun solve-invocations (invocations)
  )
