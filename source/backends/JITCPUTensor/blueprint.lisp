
(in-package :cl-waffe2/backends.jit.cpu)

;; In this device, JITCPUTensor and JITCPUScalarTensor are subject to jit-compiled.
;; Blueprint ... stores corresponding C ir node.

(defclass CPUJIT-Blueprint ()
  ((opecode :initform nil :type symbol :accessor blueprint-opecode)
   (use-vars :initform nil :type list :accessor  blueprint-use-var))
  (:documentation "
## [class] CPUJIT-Blueprint
Nodes to be involved in JIT, should extend this class.
"))

(defclass CPUJIT-Scalar-Blueprint (CPUJIT-Blueprint)
  ((opecode :initform nil :type symbol :accessor blueprint-opecode)
   (use-vars :initform nil :type list :accessor  blueprint-use-var))
  (:documentation "
## [class] CPUJIT-Scalar-Blueprint
Nodes to be involved in JIT, should extend this class.
"))

(defgeneric translate-op (opcode opast &rest args) (:documentation "
## [generic] translate-op

Return -> Instruction
"))

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; [TODO]
;; The generated C codes is merely a copy of the computation node, not optimized.
;; For efficiency, we have to fuse operations and prune unused nodes.
;;
;; Instructions that the method translate-op can return are following:
;;

;; 1. modify: A[...] += B[...]
;; 2. apply : A[...] = f(A[...])
;; 3. set   : A[...] = B[...]
;; 4. ignore: A[...]
;;
;; (2.) is composable:
;;   apply . apply = apply(apply(x))
;;

(defstruct (Instruction
	    (:constructor make-inst (inst-type function-name displace-to function-arguments)))
  "
 displace-to
   A[...]     = function-name(function-arguments)
 "
  (type inst-type :type (and keyword (member :modify :apply :set :ignore)))
  (fname function-name :type string)
  (displace-to displace-to :type AbstractTensor)
  (args function-arguments :type list))

;; TODO: Compose :apply instructions and optimize generated c codes.

