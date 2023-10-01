
(in-package :cl-waffe2/backends.jit.cpu)

(defclass CPUJIT-Blueprint ()
  ((opecode :initform nil :type symbol :accessor blueprint-opecode)
   (use-vars :initform nil :type list :accessor  blueprint-use-var))
  (:documentation "
## [class] CPUJIT-Blueprint
Nodes to be involved in JIT, should extend this class.
"))

(defgeneric translate-op (opcode opast &rest args) (:documentation "
## [generic] translate-op

Return -> Instruction
"))


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


