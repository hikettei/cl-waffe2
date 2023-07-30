
(in-package :cl-waffe2/backends.jit.cpu)

;; CPUJITTensor and ScalarTensor are subject to jit-compiled.

(defclass CPUJIT-Blueprint ()
  ((opecode :initform nil :type symbol :accessor blueprint-opecode)
   (use-vars :initform nil :type list :accessor  blueprint-use-var))
  (:documentation "
## [class] CPUJIT-Blueprint

AbstractNodes which extends this class, is recognised as `CPI-JITAble` Node by CPU-JIT-Compiler. This class possess information which is necessary for jit-compiling to cl code.
"))

(defclass CPUJIT-Scalar-Blueprint (CPUJIT-Blueprint)
  ((opecode :initform nil :type symbol :accessor blueprint-opecode)
   (use-vars :initform nil :type list :accessor  blueprint-use-var))
  (:documentation "
## [class] CPUJIT-Blueprint

AbstractNodes which extends this class, is recognised as `CPI-JITAble` Node by CPU-JIT-Compiler. This class possess information which is necessary for jit-compiling to cl code.
"))

(defgeneric translate-op (opcode opast &rest args) (:documentation "
## [generic] translate-op

Return: OpInstruction
"))

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

