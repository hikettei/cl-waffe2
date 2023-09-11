

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

  ;; Training APIs

  (:export
   #:hooker)
  
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

   #:show-backends
   #:set-devices-toplevel
   ;;#:with-cuda
   )

  (:export
   #:AbstractTrainer
   #:deftrainer
   #:optimize!
   #:zero-grads!
   #:model
   #:compiled-model
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

(in-package :cl-waffe2)

;; Export Most Basic APIs?

;;
;; :cl-waffe2/base-impl's all exported APIs
;; :cl-waffe2/distributions
;; :cl-waffe2/backends.lisp::CPUTensor, ScalarTensor etc..

;; TO ADD: cl-waffe2-repl-toplevel


;; TO ADD: Deftrainer in cl-waffe2 package

