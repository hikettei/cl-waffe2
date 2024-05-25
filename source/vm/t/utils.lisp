
(in-package :cl-waffe2/vm.test)

;; Testing Utils (change-facet ...)
;; change-facet
;; asnode
;; node->defun
;; state-dict
;; StateDict

(defsequence LazyLinear (in-features hidden-size out-features)
	     "Testing model for LinearLayer's backwards"
	     (LinearLayer in-features out-features)
	     (asnode #'!relu)
	     (LinearLayer out-features hidden-size) ;; 2th
	     (asnode #'!relu)
	     (LinearLayer hidden-size out-features) ;; 3th
	     (asnode #'!relu)
	     (asnode #'!mean))

(defun save-model-test ()
  (let ((model (build
		(call (LazyLinear 20 10 5) (make-input `(batch-size 20) :X))
		:inputs `(:X))))
    (let ((result (forward model (ax+b `(1 20) 1 0))))
      (save-weights model "./assets/model_weight_restore_tester.wf2model" :wf2model)
      (tensor-vec result))))

(defun restore-model-test (compare-to)
  (let ((model (build
		(call (LazyLinear 20 10 5) (make-input `(batch-size 20) :X))
		:inputs `(:X))))
    (load-weights model "./assets/model_weight_restore_tester.wf2model" :wf2model)
    (every #'= compare-to (tensor-vec (forward model (ax+b `(1 20) 1 0))))))

(defun save-and-restore ()
  (let ((res (save-model-test)))
    (restore-model-test res)))

(deftest save-and-restore-model-test
  (ok (save-and-restore)))


