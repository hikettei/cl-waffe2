
(in-package :cl-waffe2/backends.jit.lisp)

;; CPUJITTensor and ScalarTensor are subject to jit-compiled.

(defclass CPUJIT-Blueprint ()
  ((opecode :initform nil :type symbol :accessor blueprint-opecode)
   (use-vars :initform nil :type list :accessor  blueprint-use-var))
  (:documentation "
## [class] CPUJIT-Blueprint

AbstractNodes which extends this class, is recognised as `CPI-JITAble` Node by CPU-JIT-Compiler. This class possess information which is necessary for jit-compiling to cl code.
"))

(defclass CPUJIT-Scalar-Blueprint ()
  ((opecode :initform nil :type symbol :accessor blueprint-opecode)
   (use-vars :initform nil :type list :accessor  blueprint-use-var))
  (:documentation "
## [class] CPUJIT-Blueprint

AbstractNodes which extends this class, is recognised as `CPI-JITAble` Node by CPU-JIT-Compiler. This class possess information which is necessary for jit-compiling to cl code.
"))




