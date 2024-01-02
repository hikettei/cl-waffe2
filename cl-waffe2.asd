
(in-package :cl-user)

(defpackage :cl-waffe2-metal-asd
  (:use :cl :asdf :uiop))

(in-package :cl-waffe2-metal-asd)

(defsystem :cl-waffe2/metal-extension
  :author "hikettei"
  :licence "MIT"
  :description ""
  :pathname "cl-waffe2-metal"
  :serial t
  :depends-on (:cl-metal)
  :components ((:file "package")
	       (:file "tensor")
	       (:file "utils")
	       (:file "topi/topi")
	       (:file "topi/arithmetic")
	       (:file "topi/mathematical")))

(defpackage :cl-waffe2-simd-asd
  (:use :cl :asdf :uiop))

(in-package :cl-waffe2-simd-asd)

(defsystem :cl-waffe2/simd-extension
  :author "hikettei"
  :licence "MIT"
  :description "Utils for SIMD-Enabled Extension, CPUTensor."
  :pathname "cl-waffe2-simd"
  :serial t
  :depends-on (:cffi)
  :components ((:file "package")
	       (:file "shared-object")
	       (:file "api")))

(defpackage :cl-waffe2-asd
  (:use :cl :asdf :uiop))

(in-package :cl-waffe2-asd)

(defsystem :cl-waffe2
  :author "hikettei <ichndm@gmail.com>"
  :licence "MIT"
  :description "Programmable Deep Learning Framework for Common Lisp"
  :pathname "source"
  :serial t
  :depends-on (:cl-ppcre
	       :fiveam
	       :alexandria
	       :cffi
	       :trivia
	       :trivia.ppcre
	       :cl-randist
	       :lparallel
	       :bordeaux-threads
	       :closer-mop
	       :trivial-garbage
	       :cl-waffe2/simd-extension
	       :cl-environments
	       :numpy-file-format
	       :jonathan
	       :numcl)
  ;; TODO: Use components and split dependencies.
  :components ((:file "threads")
	       (:file "vm/generic-tensor/package")	       
	       (:file "vm/generic-tensor/conditions")
	       
	       (:file "vm/generic-tensor/default-impls")

	       ;; Load package.lisp first. (since scheduling depends on vm/nodes/package, MoveNodeTensor in base-impl/package)
	       (:file "vm/nodes/package")
	       (:file "vm/nodes/symbolic")
	       (:file "base-impl/package")
	       (:file "base-impl/attributes")

	       (:file "vm/package")
	       (:file "vm/generic-tensor/dtype")
	       (:file "vm/lazy-subscript")
	       (:file "vm/allocation")
	       ;; Iterators
	       (:file "vm/iterator/package")
	       (:file "vm/jit")
	       (:file "vm/generic-tensor/render")
	       (:file "vm/iterator/range")
	       (:file "vm/iterator/opt-loops")
	       (:file "vm/iterator/dsl")
	       (:file "vm/iterator/polyhedral")
	       
	       (:file "vm/generic-tensor/cache")
	       (:file "vm/generic-tensor/utils")
	       (:file "vm/generic-tensor/view")
	       (:file "vm/generic-tensor/call-with-view")
	       (:file "vm/generic-tensor/do-compiled-loop")
	       (:file "vm/generic-tensor/memory-pool")
	       
	       (:file "optimizers/package")
	       
	       (:file "vm/generic-tensor/acceptor" :depends-on ("vm/allocation"))
	       (:file "vm/generic-tensor/dynamic-shape")
	       (:file "vm/generic-tensor/tensor")
	       (:file "vm/generic-tensor/lut")
	       
	       (:file "vm/generic-tensor/scheduling")
	       (:file "vm/generic-tensor/state-dicts")
	       
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
	       (:file "vm/nodes/model-compiler")
	       
	       (:file "base-impl/arithmetic")
	       (:file "base-impl/fundamental")
	       (:file "base-impl/matrix-ops")
	       (:file "base-impl/reduction")
	       (:file "base-impl/mathematics")
	       (:file "base-impl/logical")
	       (:file "base-impl/transform")
	       (:file "base-impl/ir")
	       (:file "base-impl/reshapers")
	       (:file "base-impl/unfold")
	       
	       (:file "vm/ir")
	       (:file "vm/utils")
	       (:file "vm/vm")
	       (:file "vm/optimize-ir")
	       (:file "vm/compile")
	       
	       
	       (:file "backends/lisp/package")
	       (:file "backends/lisp/iterator")
	       (:file "backends/lisp/tensor")
	       (:file "backends/lisp/generic")
	       (:file "backends/lisp/arithmetic")
	       (:file "backends/lisp/mathematics")
	       (:file "backends/lisp/logical")
	       (:file "backends/lisp/matrix-ops")
	       (:file "backends/lisp/lazy")
	      
	       
	       (:file "backends/cpu/package")
	       (:file "backends/cpu/tensor")
	       
	       (:file "backends/cpu/blas")
	       (:file "backends/cpu/blas-functions")
	       (:file "backends/cpu/arithmetic")
	       (:file "backends/cpu/matrix-ops")
	       (:file "backends/cpu/mathematics")
	       (:file "backends/cpu/logical")
	       (:file "backends/cpu/im2col")
	       (:file "backends/cpu/fusion-path")
	       

	       (:file "distributions/package")
	       (:file "distributions/generic")
	       (:file "distributions/randomness")
	       (:file "distributions/sampling")
	       (:file "distributions/dense")
	       (:file "distributions/sparse")
	       (:file "distributions/ziggurat")
	       (:file "distributions/weights")

	       (:file "base-impl/utils")
	       (:file "package")
	       
	       
	       (:file "backends/lisp/wf2model")
	       (:file "backends/JITCPUTensor/package")
	       (:file "backends/JITCPUTensor/tensor")
	       (:file "backends/JITCPUTensor/compiler")
	       (:file "backends/JITCPUTensor/blueprint")	       
	       (:file "backends/JITCPUTensor/ir")
	       (:file "backends/JITCPUTensor/on-finalizing")
	       (:file "backends/JITCPUTensor/dtype")
	       (:file "backends/JITCPUTensor/foreign-function")
	       

	       (:file "backends/JITCPUTensor/impls/arithmetic")
	       (:file "backends/JITCPUTensor/impls/math")
	       
	       (:file "optimizers/defoptimizer")
	       
	       (:file "array-converter")
	       (:file "backends/lisp/im2col")
	       (:file "utils")
	       (:file "network")
	       (:file "trainer")

	       (:file "base-impl/sdiff")

	       (:file "base-impl/einsum")
	       (:file "base-impl/opt-einsum/subscripts")
	       (:file "base-impl/opt-einsum/einsum-impl")
	       (:file "base-impl/opt-einsum/utils")
	       (:file "base-impl/opt-einsum/bijective-transform")

	       (:file "nn/package")
	       (:file "nn/activation")
	       (:file "nn/regression")
	       (:file "nn/norms")
	       (:file "nn/criterion")
	       (:file "nn/recurrent")

	       (:file "nn/im2col")
	       (:file "nn/conv")
	       (:file "nn/pool")

	       (:file "optimizers/impls/sgd")
	       (:file "optimizers/impls/adam")
	       
	       (:file "viz/package")
	       (:file "viz/ast")
	       (:file "viz/dprint")
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
  :components ((:file "vm/iterator/test-suites")
	       (:file "vm/t/package")
	       (:file "vm/t/lazy-axis")
	       (:file "vm/t/utils")
	       (:file "vm/generic-tensor/t/package")
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
	       
	       (:file "nn/t/package")
	       (:file "nn/t/conv")
	       (:file "nn/t/activation")
	       (:file "nn/t/criterion")
	       (:file "nn/t/regression")	       
	       )
  :perform (test-op (o s)
		    (symbol-call :fiveam :run! :iterator-test)
		    (symbol-call :fiveam :run! :test-nodes)
		    (symbol-call :fiveam :run! :test-tensor)
		    
		    (symbol-call :fiveam :run! :base-impl-test)
		    (symbol-call :fiveam :run! :jit-lisp-test)
		    
		    (symbol-call :fiveam :run! :lisp-backend-test)
		    (symbol-call :fiveam :run! :test-backends-cpu)
		    (symbol-call :fiveam :run! :jit-cpu-test)
		    
		    (symbol-call :fiveam :run! :nn-test)
		    (symbol-call :fiveam :run! :vm-test)))


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
