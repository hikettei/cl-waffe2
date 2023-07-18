

(in-package :cl-user)

(defpackage :cl-waffe2
  (:use
   :cl
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes
   :cl-waffe2/base-impl
   :cl-waffe2/distributions
   :cl-waffe2/backends.cpu
   :cl-waffe2/backends.lisp
   :cl-waffe2/threads)
  
  ;; Facet APIs
  (:export
   #:convert-tensor-facet
   #:change-facet
   #:with-facet
   #:with-facets
   )
  
  (:export
   #:with-config
   #:with-dtype
   #:with-column-major
   #:with-row-major
   #:with-cpu
   ;;#:with-cuda
   )

  (:export
   #:AbstractTrainer
   #:deftrainer
   #:optimize!
   #:zero-grads!
   #:model
   #:set-inputs
   #:minimize!
   #:predict)

  ;; network utils
  (:export
   #:defsequence
   #:sequencelist-nth)
  
  (:export
   #:asnode
   #:call->
   ))

(defpackage :cl-waffe2-repl
  (:documentation "An playground place of cl-waffe2")
  (:use
   :cl
   :common-lisp-user
   :cl-waffe2
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.nodes
   :cl-waffe2/base-impl
   :cl-waffe2/distributions
   :cl-waffe2/backends.cpu
   :cl-waffe2/backends.lisp
   :cl-waffe2/threads
   :cl-waffe2/nn
   :cl-waffe2/optimizers))

(in-package :cl-waffe2)

;; Export Most Basic APIs?

;;
;; :cl-waffe2/base-impl's all exported APIs
;; :cl-waffe2/distributions
;; :cl-waffe2/backends.lisp::CPUTensor, ScalarTensor etc..

;; TO ADD: cl-waffe2-repl-toplevel


;; TO ADD: Deftrainer in cl-waffe2 package


