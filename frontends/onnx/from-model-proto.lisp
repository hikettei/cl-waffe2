
(in-package :cl-waffe2.frontends/onnx)

(defun from-model-proto (model-proto &key (opset) &aux (cl-onnx::*visualize* t))
  "
## [function] from-model-proto

```
(from-model-proto model-proto &key opset)
```

Converts a cl-onnx model proto into an equivalent cl-waffe2 IR.
"
  (declare (type Model-Proto model-proto))

  (let ((opset-in-model 1))
    (block detect-opsets
      (dolist (opset-id (model-proto-opset-import model-proto))
	(with-slots ((version cl-onnx::version) (domain cl-onnx::domain)) opset-id
	  ;; As per https://github.com/onnx/onnx/blob/main/docs/IR.md
	  ;; All operator sets except the default one must specify the operator version
	  (when (or (string= domain "")
		    (string= domain "ai.onnx"))
	    (setf opset-in-model version)
	    (return-from detect-opsets)))))

    (when (null opset)
      (setf opset opset-in-model))

    (when (< opset opset-in-model)
      (warn "You are overwritting the original opset ver = ~a with lower version = ~a.~%~a"
	    opset-in-model opset
	    "This might cause model conversion errors."))

    (let* ((toplevel (from-onnx (make-graph-proto-helper (model-proto-graph model-proto) opset)))
	   (composite (cl-waffe2/vm.generic-tensor:build
		       toplevel
		       :inputs
		       (map 'list #'(lambda (x) (intern (value-info-proto-name x) "KEYWORD"))
			    (graph-proto-input (model-proto-graph model-proto))))))
      (declare (type AbstractTensor toplevel))
      composite)))



