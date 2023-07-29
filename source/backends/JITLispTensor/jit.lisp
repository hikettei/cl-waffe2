
(in-package :cl-waffe2/backends.jit.lisp)

;;
;; Goals:
;;   1. Use as a model case when extending to other backends.
;;   2. FuseOps
;;   3. Eliminate this constraint: *no-grad* = t
;;   4. Focusing on composing several operations.
;;

;;
;; :cl-waffe2/backends.jit.lisp provides an example of implementing user-defined JIT Compiler
;; in cl-waffe2. This program could be applied into CUDA/Metal/C++ compiler in the future...
;;
;; For simplicity, this package compiles from cl-waffe2 into Common Lisp program.
;;

;; = [An blueprint of user-defined JIT Compiler in cl-waffe2] ===================
;;
;; 1. Goal
;;
;; Ex> (!sin (!sin (!sin x))) generates:
;;
;; (loop-with-view ...
;;          (setf (aref out ...) (sin (sin (sin i))))
;;
;; Without JIT:
;;
;; (setq out1 (loop-with-view (setf (aref out ...) (sin x))))
;; (setq out1 (loop-with-view (setf (aref out ...) (sin x))))
;; (setq out1 (loop-with-view (setf (aref out ...) (sin x)))) ...
;;

;; 2. Implementation: Embedding an additional program into compiled lisp code.
;;
;; On the end of calling of compile-forward-chain, a generic-function `on-finalizing-compiling` is invoked which users can append lisp-code as needed.
;; on-finalizing-compiling will give these informations:
;;   
;;     [TopLevel]
;;         |
;;     [SinNode1]   <- If invoked at this point...
;;         |
;;   [MoveTensorNode2]
;;         |
;;     [SinNode3]
;;         |
;;   [MoveTensorNode4]
;;         |
;;        ...
;; Info:
;;   - current-node SinNode1, to get corresponding LispJIT-Blueprint
;;   - past-variables (list TopLevel)
;;   - next-variables (list MoveTensorNode2), needed to judge whether accept node or not.
;;
;; 3. Constraints
;; Detecting the changes of devices in the nodes(e.g.: [SinNode-JITLispTensor] -> [CosNode-LispTensor]), compiler will stop tracing and compiles a kernel.
;; Detecting the changes of shapes in the nodes, compiler will stop tracing and compiles a kernel. (because complicated iteration can't be compiled in one go)
;;
;;

(defclass LispJIT-Blueprint ()
  ((opecode :initform nil :type symbol :accessor blueprint-opecode)
   (use-vars :initform nil :type list :accessor blueprint-use-var))
  (:documentation "
## [class] LispJIT-Blueprint

AbstractNodes which extends this class, is recognised as `LispJITAble` Node by Lisp-JIT-Compiler. This class possess information which is necessary for jit-compiling to cl code.
"))

(defun tensor-lisp-jit-p (tensor)
  "Returns T if the backward of tensor is a subtype of JITLispTensor"
  (let ((backward (tensor-backward tensor)))
    (subtypep (class-of backward) 'LispJIT-Blueprint)))

(defun apply-compile-p (variable next-variable)
  "Following the defition of 3., return t if there's a need to run compiling."

  ;; ViewTensorNode, PermuteTensorNode -> Compile -> ...
  ;;       ^ :device=t
  
  (or
   ;; If One of next variables are performed in different devices, or didn't exist in the first place (i.e.: is the end of nodes):
   (null next-variable)
   (funcall (compose #'not #'tensor-lisp-jit-p) next-variable)

   ;; The change of shapes is detected:
   (and (not (cl-waffe2/vm.generic-tensor::shape-equal-list (actual-shape variable) (actual-shape next-variable))))))

(defparameter *compiling-ntime-count* 0)

(defmethod on-finalizing-compiling ((current-node LispJIT-Blueprint)
				    variable
				    next-variable)
  "If the node is needed to be compiled, compile."
  (if (apply-compile-p variable next-variable)
      (progn
	(incf *compiling-ntime-count* 1)
	;;(format t "[INFO] Compiling nodes from ~a...~%" current-node)
	;; Pass these informations to invoke-compiler! function
	;; Later, compiled lisp code will be returned.
        (invoke-compiler! current-node variable next-variable))
      nil))

