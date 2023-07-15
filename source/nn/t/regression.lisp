
(in-package :cl-waffe2/nn.test)

(in-suite :nn-test)

;;
;; Comments written in JP/EN is mixed, i'm sorry.
;;


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
	     (asnode #'!relu)
	     (asnode #'!mean))


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

;; Chain Rule is Really Working?

(defun matmul-chain-test ()
  (let ((a (parameter (randn `(100 100))))
	(b (parameter (randn `(100 100))))
	(c (parameter (randn `(100 100)))))
    (proceed-backward (!sum (!matmul a (!matmul b c))))
    (and (not-zero-p a)
	 (not-zero-p b)
	 (not-zero-p c))))

(defun matmul-chain-test1 ()
  (let ((a (parameter (randn `(100 100))))
	(b (parameter (randn `(100 100))))
	(c (parameter (randn `(100 100)))))
    (proceed-backward (!matmul a (!relu (!matmul b c))))
    (and
     (not-zero-p a)
     (not-zero-p b)
     (not-zero-p c))))

(defun matmul-bias-test ()
  (let ((a (parameter (randn `(100 100))))
	(b (parameter (randn `(100 100))))
	(c (parameter (randn `(100)))))
    (proceed-backward (!sum (cl-waffe2/nn::step-linear a b c)))
    (and
     (not-zero-p a)
     (not-zero-p b)
     (every #'(lambda (x) (= x 1.0)) (tensor-vec (grad c))))))

;; Linear(ReLU(Linear(x)))
(defun linear-chain-test ()
  (let ((a1 (parameter (randn `(100 100))))
	(b1 (parameter (randn `(100 100))))
	(c1 (parameter (randn `(100))))
	(a2 (parameter (randn `(100 100))))
	(c2 (parameter (randn `(100)))))
    (proceed-backward (!mean
		       (cl-waffe2/nn::step-linear
			a2
			(!relu (cl-waffe2/nn::step-linear a1 b1 c1))
			c2)))
    (and
     (not-zero-p a1)
     (not-zero-p b1)
     (not-zero-p c1)
     (not-zero-p a2)
     (every #'(lambda (x) (= x 1e-4)) (tensor-vec (grad c2))))))

;; Doing same things with build
(defun linear-chain-test-build ()
  (let ((a1 (parameter (randn `(100 100))))
	(b1 (parameter (randn `(100 100))))
	(c1 (parameter (randn `(100))))
	(a2 (parameter (randn `(100 100))))
	(c2 (parameter (randn `(100)))))
    (let ((model (build (!mean
			 (cl-waffe2/nn::step-linear
			  a2
			  (!relu (cl-waffe2/nn::step-linear a1 b1 c1))
			  c2)))))
      (forward model)
      (backward model)
      (and
       (not-zero-p a1)
       (not-zero-p b1)
       (not-zero-p c1)
       (not-zero-p a2)
       (every #'(lambda (x) (= x 1e-4)) (tensor-vec (grad c2)))))))


(test chain-rule-test-matmul
  (is (matmul-chain-test))
  (is (matmul-chain-test1))
  (is (matmul-bias-test))
  (is (linear-chain-test))
  (is (linear-chain-test-build)))

;; ===========================================================================================
;; Bug: matmul(not(正方行列).T, any_matrix) -> Segfault
;;
;;
;;
;; Same operation with linear-composite-test-single-layer
;; Controlled Experiment:

;; これは何百回呼び出しても副作用なし
(defun linear-non-composite-test-single-layer ()
  (let ((model-weight (parameter (xavier-uniform `(100 100))))
	(model-bias   (parameter (uniform-random `(100) -0.1 0.1))))
    (let ((out (build (!mean (cl-waffe2/nn::step-linear (randn `(10 100))
							model-weight
							model-bias)))))
      (forward out)
      (backward out)
      (and
       (not-zero-p model-weight)
       (every #'(lambda (x) (= x 0.001)) (tensor-vec (grad model-bias)))))))

;; Setting construct-backward? = nil -> it is working.
;; => Therefore, system-lazy-save-for-backward do not contribute to this problem.
(defun linear-non-composite-test-single-layer-no-bw ()
  (let ((model-weight (parameter (xavier-uniform `(100 100))))
	(model-bias   (parameter (uniform-random `(100) -0.1 0.1))))
    (let ((out (build (!mean (cl-waffe2/nn::step-linear (randn `(10 100))
							model-weight
							model-bias))
		      :construct-backward? nil)))
      (forward out))))


;; Multiple call <- No Side Effects???
(test linear-simple-layer
  (is (linear-non-composite-test-single-layer))
  (is (linear-non-composite-test-single-layer))
  (is (linear-non-composite-test-single-layer))
  (is (linear-non-composite-test-single-layer-no-bw)))

;; Known Issue: Second call of this got invaild.
;; 問題は、Compositeを用いてモデルを初期化した時に、二回目以降のコンパイルがおかしくなる

;; Cache関数が悪いか、メモリプールが悪いか
;; Occurs at ADDNODECPUTENSOR-VM-FUNCTION
;; build関数とstep-linearを用いた同じノードは動く
;; -> 多分Compositeを介してるから発生してる？
;; -> Compositeが何かの副作用を・・・
(defun linear-composite-test-single-layer ()
  (let ((model (LinearLayer 100 10)))
    (let ((model (build (!mean (call model (randn `(10 100)))))))
      (forward model))))

;; Linear合成しても問題は同じ
(defun linear-composite-test-two-layer ()
  (let ((model  (LinearLayer 100 10))
	(model1 (LinearLayer 10 3)))
    (let ((compiled-model (build (!mean (call model1 (call model (randn `(10 100))))))))
      (forward compiled-model))))

;; 原因は二つあって
;; 遅延評価!tがうまいタイミングでAllocされなかった（解決済み）
;; Composite使ったmatmulが二回目で失敗（原因なぞ）

;; no-grad = Tだと発生しない (?)
;; no-grad = NILだと発生する (?)

;; save-for-backwardのMoveTensorが原因であるはず。。。 (違った)
;; -> system-lazy-save-for-backwardはちゃんと動いている

;; 多分どっちか：
;; -> Memory-PoolにCacheされたInputTensorをPermuteするのが悪いのか
;; -> Backwardの関数をコンパイルしてる途中で何らかの副作用があるのか？
;; -> Cacheされた関数？

;; Knwon Issue: 二回目のCallでmatmulに失敗する？
(test linear-layer-test-forward
  (is (linear-composite-test-single-layer))
  (is (linear-composite-test-single-layer))
  (is (linear-composite-test-single-layer)))


;; Test Forward
(test linear-composed-layer-test-forward
  (is (linear-composite-test-two-layer))
  (is (linear-composite-test-two-layer))
  (is (linear-composite-test-two-layer)))

;; Regardless of composite use, it occurs

(defmacro with-model-parameters ((bind model) &body body)
  `(let ((,bind (nodevariables-parameters
		 (compiled-variables ,model))))
     ,@body))

;; Simple Case:
;; Adjustable-Symbol <- None
;; static-node       <- None
;;
;; Only using pure features in cl-waffe2
;; OK
(defun linearlayer-backward-test ()
  (with-memory-pool
    (let* ((model (LinearLayer-Sequence 100 50 10))
	   (model (build (call model (uniform-random `(10 100) -0.01 0.01))
			 :compile-mode :default)))
      (forward model)
      (backward model)
      (with-model-parameters (params model)
	;;(loop for p in params
	;;      do (print (grad p)))
	(every #'not-zero-p params)))))

(test linear-backward-test-only-with-principle-features
  (is (linearlayer-backward-test))
  (is (linearlayer-backward-test))
  (is (linearlayer-backward-test))
  )

;; Second Case:
;; Adjustable-Symbol <- None
;; static-node       <- T
;;
;; Using criterion
;; Here's not working...
;; Once the form below is called, memory-pool is destructed.
(defun linearlayer-backward-test-with-criterion ()
  (let* ((model (LinearLayer-Sequence1 100 50 10))
	 (model (build (!mean
			(softmax-cross-entropy
			 (call model (randn `(10 100)))
			 (randn `(10 10))))
		       :compile-mode :default)))
    (forward model)
    (backward model)
    (with-model-parameters (params model)
      ;;(loop for p in params
      ;;	    do (print (grad p)))
      (every #'not-zero-p params))))

(test linearlayer-backward-with-criterlion
  (is (linearlayer-backward-test-with-criterion)))
	     

;; これからデバッグすること：
;; Linearを重ねたときにBackwardできてるか？（多分おK)
;; Matmulを複数回呼び出してInvaild...
;; memory-poolをテストする

