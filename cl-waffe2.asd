
(in-package :cl-user)

(defpackage :cl-waffe2-asd
  (:use :cl :asdf :uiop))

(in-package :cl-waffe2-asd)

(defsystem :cl-waffe2
  :author "hikettei"
  :licence "MIT"
  :description "Deep Learning Framework"
  :pathname "source"
  :serial t
  :depends-on (:cl-ppcre :fiveam :alexandria :cffi :numcl :lparallel)
  :components ((:file "threads")
	       (:file "vm/generic-tensor/package")
	       (:file "vm/generic-tensor/conditions")
	       (:file "vm/generic-tensor/utils")
	       (:file "vm/generic-tensor/dtype")
	       
	       (:file "vm/generic-tensor/render")
	       (:file "vm/generic-tensor/default-impls")

	       ;; Load package.lisp first. (since scheduling depends on vm/nodes/package, MoveNodeTensor in base-impl/package)
	       (:file "vm/nodes/package")
	       (:file "base-impl/package")
	       (:file "vm/generic-tensor/view")
	       

	       (:file "vm/generic-tensor/tensor")
	       (:file "vm/generic-tensor/acceptor")
	       
	       (:file "vm/generic-tensor/scheduling")
	       
	       (:file "vm/nodes/shape")
	       (:file "vm/nodes/node")
	       (:file "vm/nodes/conditions")
	       (:file "vm/nodes/defnode")
	       (:file "vm/nodes/defmodel")
	       (:file "vm/nodes/utils")

	       
	       (:file "base-impl/arithmetic")
	       (:file "base-impl/fundamental")
	       (:file "base-impl/matrix-ops")
	       (:file "base-impl/reduction")
	       (:file "base-impl/mathematics")
	       (:file "base-impl/logical")

	       (:file "backends/lisp/package")
	       (:file "backends/lisp/tensor")
	       (:file "backends/lisp/generic")
	       (:file "backends/lisp/arithmetic")
	       (:file "backends/lisp/mathematics")
	       (:file "backends/lisp/logical")
	       
	       (:file "backends/cpu/package")
	       (:file "backends/cpu/tensor")
	       
	       (:file "backends/cpu/blas")
	       (:file "backends/cpu/blas-functions")
	       (:file "backends/cpu/arithmetic")
	       (:file "backends/cpu/matrix-ops")

	       (:file "distributions/package")
	       (:file "distributions/generic")
	       (:file "distributions/randomness")
	       (:file "distributions/sampling")
	       (:file "distributions/dense")
	       (:file "distributions/sparse")
	       (:file "distributions/ziggurat")

	       (:file "nn/package")
	       (:file "nn/regression")
	       (:file "optimizers/package")

	       (:file "package")
	       (:file "utils")

	       (:file "viz/package")
	       (:file "viz/ast")
	       
	       )
  :in-order-to ((test-op (test-op cl-waffe2/test))))

(defpackage :cl-waffe2-test
  (:use :cl :asdf :uiop))

(in-package :cl-waffe2-test)

(defsystem :cl-waffe2/test
  :author "hikettei"
  :licence "MIT"
  :description "Tests for cl-waffe2"
  :serial t
  :pathname "source"
  :depends-on (:cl-waffe2 :fiveam)
  :components ((:file "vm/generic-tensor/t/package")
	       (:file "vm/generic-tensor/t/forward")
	       (:file "vm/generic-tensor/t/backward")
	       (:file "vm/generic-tensor/t/view")
	       (:file "vm/generic-tensor/t/optimize")
	       
	       (:file "vm/nodes/t/package")
	       (:file "vm/nodes/t/parser")
	       (:file "vm/nodes/t/shape")
	       (:file "vm/nodes/t/nodes")

	       (:file "base-impl/t/package")
	       (:file "base-impl/t/reduction")
	       (:file "base-impl/t/arithmetic")
	       (:file "base-impl/t/mathematical")
	       (:file "base-impl/t/apis")

	       (:file "backends/lisp/t/package")
	       
	       (:file "backends/cpu/t/package")
	       (:file "backends/cpu/t/arithmetic")
	       
	       )
  :perform (test-op (o s)
		    (symbol-call :fiveam :run! :test-nodes)
		    (symbol-call :fiveam :run! :test-tensor)
		    (symbol-call :fiveam :run! :base-impl-test)
		    (symbol-call :fiveam :run! :lisp-backend-test)
		    (symbol-call :fiveam :run! :test-backends-cpu)
		    
		    ))


(defpackage :cl-waffe2-docs-asdf
  (:use :cl :asdf :uiop))

(in-package :cl-waffe2-docs-asdf)

(defsystem :cl-waffe2/docs
  :author "hikettei"
  :licence "MIT"
  :description "Documentation Generator for cl-waffe2"
  :serial t
  :pathname "docs"
  :depends-on (:codex :cl-ppcre)
  :components ((:file "package")
	       (:file "introduction/overview")
	       (:file "apis/reference")))


