
(in-package :cl-waffe2/vm)


;; Provides a compiler from cl-waffe2 IR -> Instruction Sequence

;; cl-waffe2 IR
;;     - Tree-structure
;;     - represented by AbstractTensor(S expression + Shape + CLOS Class)
;;     - or sometimes composable call-with-view

;; Instruction Sequence
;;    - Flatten
;;    - represetned by lambda expression

(defparameter *vm-compile-option* :default)

(declaim (ftype (function (AbstractTensor) (or null WFInstruction)) ir->instruction))
(defun ir->instruction (tensor)
  "Reading a IR of tensor, the function returns a corresponding instruction"
  (declare (type AbstractTensor tensor))

  (cond
    ((null (tensor-backward tensor))
     ;; Has reached out the end of nodes.
     nil)
    (T
     (make-wfop
      (apply
       #'find-cached-function
       (statecontainer-forward-out-form (tensor-state tensor))
       (cl-waffe2/vm.generic-tensor::compile-option-form *vm-compile-option*)
       (tensor-variables tensor))
      tensor
      (tensor-backward tensor)
      (tensor-variables tensor)))))

;;
;; Avoid duplicate compilation:
;;
;; (!sin x (!copy x))
;;

(defun compile-into-vm (toplevel)
  (let ((instruction-seq))
    (labels ((explore-node (tensor &key (stop-me nil) (prev-var nil))
	       (let ((out (ir->instruction tensor)))
		 (when out
		   (push out instruction-seq)))
	       (when (not stop-me)
		 (dolist (var (tensor-variables tensor))
		   ;; 右側の計算ノードが左側を内包してるみたいな場合がある
		   ;; それは場外したい・・・
		   (explore-node var :stop-me (detach-p var) :prev-var tensor)))))
      (explore-node toplevel :prev-var nil)
      instruction-seq)))


;;(let ((a (!tan (randn `(3 3)))))
;; (compile-into-vm
;;  (!sin a :-> (!cos a))))

;; tan must appeared at once in the compiled code..

;; (let ((a (!tan (randn `(3 3)))))
;;		(compile-into-vm
;;		 (!sin (!sin a :-> a) :-> a))
;;		)
