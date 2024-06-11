
;; [WIP] Feature roadmap on frontends.
;;
;;  frontend/onnx (From ONNX To cl-waffe2 translator)
;;  frontend/qnn  (Quantized Neural Network Op Supports (basically relies on JITCompiler))
;;  frontend/backends
;;  frontend/backends/cuda (Additional backends which requires more dependency, should be placed at frontend)
;;

(asdf:defsystem :cl-waffe2.frontends
  :description "A set of frontend toolkits for cl-waffe2"
  :author "hikettei <ichndm@gmail.com>"
  :Licence "MIT")

(asdf:defsystem :cl-waffe2.frontends/onnx
  :description "ONNX Graph Translator"
  :author      "hikettei <ichndm@gmail.com>"
  :licence     "MIT"
  :pathname   "frontends/onnx"
  :depends-on ("cl-onnx" "alexandria")
  :components ((:file "package")
	       (:file "utils")
	       (:file "opset")
	       (:file "from-model-proto")))

;; Other frontend system follows...
;;(asdf:defsystem :cl-waffe2.frontends/qnn
;;  :description "[WIP] Quantization Tools"
;;  :author      "hikettei <ichndm@gmail.com>"
;;  :licence     "MIT")
