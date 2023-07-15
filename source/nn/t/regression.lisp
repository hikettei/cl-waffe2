
(in-package :cl-waffe2/nn.test)

(in-suite :nn-test)


;; Plus: Diff MLP

;; Memo: build <- unsafe.


;; Known Issue: Computing the backwards of sequence of LinearLayer,
;; Some weights of layers (esp, 2th~3th), will become zero.
;; The assumption is that acceptor.lisp contributes to this problems.

(defsequence LinearLayer-Sequence (in-features hidden-size out-features)
	     "Testing model for LinearLayer's backwards"
	     (LinearLayer in-features out-features)
	     (asnode #'!relu)
	     (LinearLayer out-features hidden-size) ;; 2th
	     (asnode #'!relu)
	     (LinearLayer hidden-size out-features) ;; 3th
	     (asnode #'!sigmoid)
	     (asnode #'!sum))


(defsequence LinearLayer-Sequence1 (in-features hidden-size out-features)
	     "Testing model for LinearLayer's backwards"
	     (LinearLayer in-features out-features)
	     (asnode #'!relu)
	     (LinearLayer out-features hidden-size) ;; 2th
	     (asnode #'!relu)
	     (LinearLayer hidden-size out-features) ;; 3th
	     )

(defun not-zero-p (tensor)
  (some #'(lambda (x) (not (= x 0))) (tensor-vec (grad tensor))))

(defmacro with-model-parameters ((bind model) &body body)
  `(let ((,bind (nodevariables-parameters
		 (compiled-variables ,model))))
     ,@body))

;; Simple Case:
;; Adjustable-Symbol <- None
;; static-node       <- None
;;
;; Only using pure features in cl-waffe2.
(defun linearlayer-backward-test ()
  (let* ((model (LinearLayer-Sequence 100 50 10))
	 (model (build (call model (randn `(10 100))))))
    (forward model)
    (backward model)
    (with-model-parameters (params model)
      (every #'not-zero-p params))))

(test linear-backward-test-only-with-principle-features
  (is (linearlayer-backward-test)))

;; Second Case:
;; Adjustable-Symbol <- None
;; static-node       <- T
;;
;; Using criterion
;; Here's not working...
;; Once the form below is called, memory-pool is destructed.
(defun linearlayer-backward-test-with-criterion ()
  (with-no-grad
  (let* ((model (LinearLayer-Sequence1 100 50 10))
	 (model (build (!mean
			(softmax-cross-entropy
			 (call model (randn `(10 100)))
			 (randn `(10 10))))
		       :compile-mode :default)))
    (print (forward model))
   ;; (backward model)
    (with-model-parameters (params model)
      (print params)
      (every #'not-zero-p params)))))

(test linearlayer-backward-with-criterion
  (is (linearlayer-backward-test-with-criterion)))
	     
