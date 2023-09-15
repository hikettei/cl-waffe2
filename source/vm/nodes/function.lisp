
(in-package :cl-waffe2/vm.nodes)

;; I'm depcrecated with Composite-Function
;; Should be deleted in the future release:


;; The file function.lisp provides a system on interacting Lazy-Evaluated Nodes and Toplevel functions:
;;
;; Composite <-> Function
;; Node      <-> Function
;;

;; One composite -> a single defun form.
;; In order to implemenet "GENERIC" behaviour, we need to wrap composite->defun by higher-order function.

(defparameter *jaot-shape-error-detection* T "
## [parameter] `*jaot-shape-error-detection*`
(TODO)
")

(defun eliminate-undetermined-size (tensor)
  (let* ((shape (cl-waffe2/vm.generic-tensor::translate-adjustable-shape (actual-shape tensor)))
	 (out (make-input  shape nil
			   :create-from tensor
			   :dtype (dtype tensor)
			   :order (order tensor)
			   :scalar-p (scalar-p tensor)))
	 (broadcasted-p)
	 (broadcasts (loop for size in (cl-waffe2/vm.generic-tensor::translate-adjustable-shape (shape tensor))
			   for view in (tensor-view tensor)
			   if (eql :broadcast (cl-waffe2/vm.generic-tensor::viewtype (cl-waffe2/vm.generic-tensor::force-list view)))
			     collect (and
				      (setq broadcasted-p t)
				      `(:broadcast ,size))
			   else
			     collect t))
	 (out (if broadcasted-p
		  (apply #'cl-waffe2/vm.generic-tensor::view out broadcasts)
		  out)))
    
    (setf (cl-waffe2/vm.generic-tensor::tensor-initial-offset out) (cl-waffe2/vm.generic-tensor::tensor-initial-offset tensor)
	  (tensor-vec out) (cl-waffe2/vm.generic-tensor::vec tensor))
    out))

;; [Experimental] Aot Shape-Error Detection?
;;  1. This setting should be disabled with parameters
;;  2. forward earlier compilingで有効にする <- dispatchingしない defnode宣言だけで十分
;;     ^ eval-when (:compile-toplevel) ... でHash-Tableに記録しておく + dtypeの検査などを省く + not-found errorにしないように
;;  3. ScalarTensor/dtypeの検査がしたい・・・
;;  4. 余計なcompileと明示されてない関数の実行が走るから、node->defun (トレスが何回も走る前提) で やる

;;  Printing the position:
;; Still Experimental and a lot of challenges are remained:
;;  ScalarTensor/Dtype
;; Inserts AoT Shape-Error Inspection Code

(defmacro with-aot-shape-error-detection ((&key
					     (in-names nil)
					     (in-shapes nil)
					     (out-names nil)
					     (out-shapes nil))
					  &body
					    body)
  "## [macro] with-aot-shape-error-detection
Wrap the body you want to enable aot shape-error detection"
  ;; ~を含むと未決定になる
  ;; where scal = 1 ... regard as ScalarTensor?
  ;; ScalarTensorとそれ以外の区別しない？
  (eval-when (:execute)
    (if (or (not *jaot-shape-error-detection*)
	    (some #'(lambda (shape)
		      (some #'(lambda (x) (symbol-eq x '~))
			    shape))
		  in-shapes))
	body
	(let* ((in-tensors  (loop for name in in-names
			 	  for shape in in-shapes
				  collect
				  (make-input shape name)))
	       ;; [TODO] Implement :forward but JAOT Mode
	       ;; [TODO] Just tracing the transimissions of forward
	       ;; [TODO] Should be displayed as just a warning, not an error
	       ;; [TODO]
	       (outs (multiple-value-list (apply #'call (cl-waffe2::asnode (compile nil `,(car body))) in-tensors))))
	  ;; out-shapesが~を含むとき -> Ignore
	  (assert (every #'equal out-shapes
			 (map 'list #'shape outs))
		  nil
		  "AOT Shape Error: Out is invaild. ~a ~a" out-shapes (map 'list #'shape outs))
	  `(progn ,@body)))))

;;(use-package :cl-waffe2/base-impl)

(defun aot-error-test ()
  (with-aot-shape-error-detection (:in-names (:A :B)
				   :in-shapes ((x y) (x y))
				   :out-names (:OUT)
				   :out-shapes ((1 1)))
    (lambda (x y)
      (cl-waffe2/base-impl:!mean (cl-waffe2/base-impl:!add y (cl-waffe2/base-impl:!matmul x (cl-waffe2/base-impl:!t x)))))))



