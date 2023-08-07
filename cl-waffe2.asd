
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
  :depends-on (:cl-ppcre
	       :fiveam
	       :alexandria
	       :cffi
	       :cl-randist
	       :lparallel
	       :bordeaux-threads
	       :closer-mop
	       :optima
	       :trivial-garbage)
  ;; TODO: Use components and split dependencies.
  :components ((:file "threads")
	       (:file "vm/generic-tensor/package")
	       (:file "vm/generic-tensor/conditions")
	       
	       (:file "vm/generic-tensor/dtype")
	       
	       
	       (:file "vm/generic-tensor/render")
	       (:file "vm/generic-tensor/default-impls")

	       ;; Load package.lisp first. (since scheduling depends on vm/nodes/package, MoveNodeTensor in base-impl/package)
	       (:file "vm/nodes/package")
	       (:file "base-impl/package")
	       (:file "vm/generic-tensor/cache")
	       (:file "vm/generic-tensor/utils")
	       (:file "vm/generic-tensor/view")
	       (:file "vm/generic-tensor/call-with-view")
	       (:file "vm/generic-tensor/memory-pool")

	       (:file "optimizers/package")
	       
	       (:file "vm/generic-tensor/acceptor")
	       (:file "vm/generic-tensor/tensor")
	       (:file "vm/generic-tensor/interpreter")
	       (:file "vm/generic-tensor/lut")
	       
	       (:file "vm/generic-tensor/scheduling")

	       (:file "vm/nodes/shape-error")
	       (:file "vm/nodes/shape")
	       (:file "vm/nodes/broadcast")
	       (:file "vm/nodes/node")
	       (:file "vm/nodes/conditions")
	       (:file "vm/nodes/defnode")
	       (:file "vm/nodes/defmodel")
	       (:file "vm/nodes/utils")
	       (:file "vm/nodes/function")
	       (:file "vm/nodes/static-node")

	       
	       (:file "base-impl/arithmetic")
	       (:file "base-impl/fundamental")
	       (:file "base-impl/matrix-ops")
	       (:file "base-impl/reduction")
	       (:file "base-impl/mathematics")
	       (:file "base-impl/logical")
	       (:file "base-impl/transform")
	       (:file "base-impl/ir")
	       (:file "base-impl/reshapers")
	       
	       

	       (:file "backends/lisp/package")
	       (:file "backends/lisp/tensor")
	       (:file "backends/lisp/generic")
	       (:file "backends/lisp/arithmetic")
	       (:file "backends/lisp/mathematics")
	       (:file "backends/lisp/logical")
	       (:file "backends/lisp/matrix-ops")
	       
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
	       (:file "distributions/weights")

	       (:file "base-impl/utils")
	       
	       (:file "backends/JITLispTensor/package")
	       (:file "backends/JITLispTensor/compiler")
	       (:file "backends/JITLispTensor/tensor")
	       (:file "backends/JITLispTensor/jit")
	       (:file "backends/JITLispTensor/delayed-node-impls")

	       (:file "package")
	       (:file "backends/JITCPUTensor/package")
	       (:file "backends/JITCPUTensor/tensor")
	       (:file "backends/JITCPUTensor/blueprint")
	       (:file "backends/JITCPUTensor/ir")
	       (:file "backends/JITCPUTensor/on-finalizing")
	       (:file "backends/JITCPUTensor/dtype")
	       (:file "backends/JITCPUTensor/compiler")
	       (:file "backends/JITCPUTensor/foreign-function")
	       

	       (:file "backends/JITCPUTensor/impls/arithmetic")
	       (:file "backends/JITCPUTensor/impls/math")
	       
	       (:file "optimizers/defoptimizer")
	       
	       (:file "array-converter")
	       (:file "utils")
	       (:file "network")
	       (:file "trainer")

	       (:file "base-impl/einsum")
	       (:file "base-impl/opt-einsum/subscripts")
	       (:file "base-impl/opt-einsum/einsum-impl")
	       (:file "base-impl/opt-einsum/utils")
	       (:file "base-impl/opt-einsum/bijective-transform")

	       (:file "nn/package")
	       (:file "nn/activation")
	       (:file "nn/regression")
	       (:file "nn/criterion")
	       (:file "nn/recurrent")

	       (:file "nn/im2col")
	       (:file "nn/conv")
	       (:file "nn/pool")

	       (:file "optimizers/impls/sgd")
	       (:file "optimizers/impls/adam")
	       
	       (:file "viz/package")
	       (:file "viz/ast")
	       (:file "cl-waffe2-repl")
	       
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
	       (:file "vm/nodes/t/composite")
	       (:file "vm/nodes/t/static-node")

	       (:file "base-impl/t/package")
	       (:file "base-impl/t/reduction")
	       (:file "base-impl/t/arithmetic")
	       (:file "base-impl/t/mathematical")
	       (:file "base-impl/t/apis")
	       
	       (:file "backends/cpu/t/package")
	       (:file "backends/cpu/t/arithmetic")

	       (:file "backends/lisp/t/package")

	       (:file "backends/JITCPUTensor/t/package")
	       (:file "backends/JITCPUTensor/t/jit")
	       
	       (:file "backends/JITLispTensor/t/package")
	       (:file "backends/JITLispTensor/t/compiler")

	       (:file "nn/t/package")
	       (:file "nn/t/conv")
	       (:file "nn/t/activation")
	       (:file "nn/t/criterion")
	       (:file "nn/t/regression")
	       
	       )
  :perform (test-op (o s)
		    (symbol-call :fiveam :run! :jit-lisp-test)
		    
		    (symbol-call :fiveam :run! :test-nodes)
		    (symbol-call :fiveam :run! :test-tensor)
		    (symbol-call :fiveam :run! :base-impl-test)

		    (symbol-call :fiveam :run! :jit-cpu-test)
		    (symbol-call :fiveam :run! :lisp-backend-test)
		    (symbol-call :fiveam :run! :test-backends-cpu)
		    (symbol-call :fiveam :run! :nn-test)))


(defpackage :cl-waffe2-docs-asdf
  (:use :cl :asdf :uiop))

(in-package :cl-waffe2-docs-asdf)

(defsystem :cl-waffe2/docs
  :author "hikettei"
  :licence "MIT"
  :description "Documentation Generator for cl-waffe2"
  :serial t
  :pathname "docs"
  :depends-on (:cl-ppcre)
  :components ((:file "package")
	       (:file "apis/reference")
	       (:file "apis/distributions")
	       (:file "apis/nodes")
	       (:file "apis/generic-tensor")
	       (:file "apis/nn")
	       (:file "apis/optimizers")
	       ))



(defpackage :cl-waffe2-bench-asdf
  (:use :cl :asdf :uiop))

(in-package :cl-waffe2-bench-asdf)

(defsystem :cl-waffe2/benchmark
  :author "hikettei"
  :licence "MIT"
  :description "Benchmark on cl-waffe2"
  :serial t
  :pathname "benchmarks"
  :depends-on (:cl-ppcre)
  :components ((:file "package")
	       (:file "benchmark")
	       (:file "element-wise")
	       (:file "compose")
	       (:file "model")
	       (:file "reverse")
	       (:file "profile")))

