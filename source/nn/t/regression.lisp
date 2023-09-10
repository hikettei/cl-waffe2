
(in-package :cl-waffe2/nn.test)

(in-suite :nn-test)

;;
;; Comments written in JP/EN is mixed, i'm sorry.
;;


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

(defun matmul-chain-test-smol ()
  (let ((a (parameter (ax+b `(3 3) 0 2)))
	(b (parameter (ax+b `(3 5) 0 3)))
	(c (parameter (ax+b `(5 3) 0 4))))
    (proceed-backward (!sum (!matmul a (!matmul b c))))
    (print (grad a))
    (print (grad b))
    (print (grad c))
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
;; Bug: matmul(not(正方行列).T, any_matrix) -> Segfault (Now it's FIXED)
;;
;;
;;
;; Same operation with linear-composite-test-single-layer
;; Controlled Experiment:

;; The Problem is that:
;; No matter how many times I invoke this function, there's no side effects.
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

;; But combined with composite, the second call of matmul will produce shape-error???
(defun linear-composite-test-single-layer ()
  (let ((model (LinearLayer 5 2)))
    (let ((model (build (!mean (call model (randn `(2 5)))))))
      (forward model)
      ;; (forward model) <- is working
      )))

;; Even when composed, the problem remains
(defun linear-composite-test-two-layer ()
  (let ((model  (LinearLayer 5 3))
	(model1 (LinearLayer 3 2)))
    (let ((compiled-model (build (!mean (call model1 (call model (randn `(1 5))))))))
      (forward compiled-model))))

;; 原因は二つあった?
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

;; ugokan
(test linear-layer-test-forward
  (is (linear-composite-test-single-layer))
  (is (linear-composite-test-single-layer))
  (is (linear-composite-test-single-layer)))


;; ugokan
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
  (progn;with-memory-pool
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
  (is (linearlayer-backward-test-with-criterion))
  ;; Is the cached function, works well?
  (is (linearlayer-backward-test-with-criterion)))
	     

;; これからデバッグすること：
;; Traceのネストが深い理由
;; -> requires-grad=tのテンソルを作成した時毎回コンパイルしてたから（修正済み）

;; Linearを重ねた時に中間層のMatmulのgradが0 -> !tでtensor-vecしてないから (修正済み）

;; 層を重ねても動いてる

;; memory-poolをテストする
;; Cacheされた関数テスト <- ignoreのやつほんとに有効になってる？

;; どこのマクロの展開式がダメ？
;; 関数コンパイルして一回目は動作、二回目以降は動かない
;; forwardを何回呼び出しても変わらない 関数を何回呼び出すかである

;; with-no-gradで呼び出す分には副作用が発生しない。
;; やっぱりPermute*が原因だと思う。
;;

;; 二回目のXの入力が5 2 -> 2 5になってるけどどうして？
;; (call model ExistTensor) するとX
;; (call model Copy) nara Ok

;; **randnのShapeをCopyしたら動いた WHY??**
;; `(1 2 ...) <- 参照渡しだっけ？
(defun matmul-bug-case ()
  (let ((model (LinearLayer 5 2)))
    (let ((model (build (!mean (call model (randn `(2 5))))
			:compile-mode :safety)))
      (forward model)
      (forward model)
      )))

(defmodel (Softmax-Model (self)
	   :where (X[~] -> [~])
	   :on-call-> ((self x)
		       (declare (ignore self))
		       (let* ((x1 (!sub x (!mean x  :axis 1 :keepdims t)))
                              (z  (!sum   (!exp x1) :axis 1 :keepdims t)))
                         (!div (!exp x1) z)))))

;; It is now working
(defun softmax-same-case? ()
  (let ((model (softmax-model)))
    (let ((model (build (!mean (call model (randn `(2 5)))))))
      (forward model))))

(test softmax-no-side-effect-call-of-composite
  (is (softmax-same-case?))
  (is (softmax-same-case?))
  (is (softmax-same-case?)))

(defun fw-change-shape-test ()
  (let ((model (build (call (LinearLayer-Sequence 10 5 2)
			    (make-input `(batch-size 10) :X))
		      :compile-mode :safety)))
    (set-input model :X (randn `(10 10)))
    (forward model)
    (set-input model :X (randn `(20 10)))
    (forward model)
    (set-input model :X (randn `(5 10)))
    (forward model)
    model))

(test forward-with-different-shape
  (is (fw-change-shape-test)))

(defun fw-and-bw-test ()
  (let* ((linear (LinearLayer-Sequence 10 5 2))
	 (model (build (call linear
			     (make-input `(batch-size 10) :X)))))
    (set-input model :X (ax+b `(10 10) 0.0 1.0))
    (forward model)
    (backward model)
    
    ;;(with-model-parameters (params model)
    ;;  (loop for p in params
;;	    do (print (grad p))
;;	    do (print (tensor-vec (grad p)))))
    (forward model)
    (backward model)
  ;;  (with-model-parameters (params model)
    ;;  (loop for p in params
;;	    do (print (grad p))
;;	    do (print (tensor-vec (grad p)))))
    (forward model)
    (backward model)
  ;;  (with-model-parameters (params model)
    ;;  (loop for p in params
;;	    do (print (grad p))
;;	    do (print (tensor-vec (grad p)))))

    (forward model)
    (backward model)
    ))

(defun fw-and-bw-test-criterion ()
  (let* ((model (LinearLayer-Sequence1 100 50 10))
	 (model (build (!mean
			(softmax-cross-entropy
			 (call model (make-input `(batch-size 100) :X))
			 (make-input `(batch-size 10) :Y)))
		       :compile-mode :safety)))
    (set-input model :X (randn `(10 100)))
    (set-input model :Y (randn `(10 10)))

    (forward model)
    (backward model)

    (with-model-parameters (param model)
      (loop for p in param
	    do (grad p)))

    ;; Segfault here.
    (forward model)
    (backward model)

    (with-model-parameters (param model)
      (loop for p in param
	    do  (grad p)))
    T))

(test multiple-time-call-of-compiled-model
  (is (fw-and-bw-test))
  (is (fw-and-bw-test-criterion))
  )


;; Gradients are decayed well?


(defsequence Simple-MLP (in-features hidden-dim)
	     (LinearLayer in-features hidden-dim t)
	     (asnode #'!sigmoid)
	     (LinearLayer hidden-dim 1 t))

;; Naming:... set-input VS set-inputs
;; allow: (forward self)
;; make trainer forwardable

(deftrainer (MLPTrainer (self in-features hidden-dim &key (lr 1e-1))
	     :model (Simple-MLP in-features hidden-dim)
	     :optimizer (cl-waffe2/optimizers:SGD :lr lr)
	     :compile-mode :fastest
	     :build ((self)
		     (MSE
		      (make-input `(batch-size 1) :TrainY)
		      (call (model self) (make-input `(batch-size ,in-features) :TrainX))))
	     
	     :set-inputs ((self x y)
			  (set-input (compiled-model self) :TrainX x)
			  (set-input (compiled-model self) :TrainY y))
	     :minimize! ((self)
			 (zero-grads! (compiled-model self))
			 (let ((loss (forward (compiled-model self))))
			   (backward  (compiled-model self))
			   (optimize! (compiled-model self))
			   (vref loss 0)))			 
	     :predict ((self x)
		       (call (model self) x))))


(defun grad-decay-test (&key
			  (batch-size 100)
			  (iter-num 3000))
  (let* ((X (proceed (!sin (ax+b `(,batch-size 100) 0.01 0.1))))
 	 (Y (proceed (!cos (ax+b `(,batch-size 1)   0.01 0.1))))
	 (trainer (MLPTrainer 100 10 :lr 1e-3))
	 (first)
	 (end))
    
    (set-inputs trainer X Y)
    (loop for nth-epoch fixnum upfrom 0 below iter-num
	  do (let ((out (minimize! trainer)))
	       (if (null first) (setq first out))
	       (setq end out)))
    (> first end)))

(test grad-decay-test
  (is (grad-decay-test)))

(test grad-decay-cached-test
  (is (grad-decay-test)))

