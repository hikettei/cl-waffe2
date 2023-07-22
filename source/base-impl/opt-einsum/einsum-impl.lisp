

(in-package :cl-waffe2/base-impl)

;; ============================================================
;; Referenced:
;; https://zenn.dev/lotz/articles/b77c3434fa9451fcc927
;; https://zenn.dev/termoshtt/articles/einsum-derive
;; 
;; https://github.com/dgasmith/opt_einsum
;; https://dgasmith.github.io/opt_einsum/paths/introduction/
;; https://dgasmith.github.io/opt_einsum/paths/optimal_path/
;; https://discuss.pytorch.org/t/automatic-differentation-for-pytorch-einsum/112504
;; ============================================================

;;;; 

;; ================================================================
;; einsum.lisp provides the einsten notation:
;;
;; (dotimes (i 100)
;;   (dotimes (j 100)
;;     (setf (aref out i j) (+ (aref x i j) (aref y j i)))))
;;
;; -> Just lining up the corresponding subscripts of aref:
;;  (%einsum X[i j] Y[j i] -> OUT[i j])
;; The result is the same as first one.
;; ================================================================



;; ==================================================================
;; APIs:
;;
;; [macro] %einsum
;; Example:
;;   (%einsum A[i j] -> B[i j])
;; [macro] %implict-einsum
;;   (%implict-einsum A[i j]) ;; the macro adds -> [~] form implictly
;; [function] !einsum
;;   (!einsum ...)
;; ==================================================================



;; To simplify the einsum rule, cl-waffe2 follows the rule below:

;; ===============================================
;; Base syntax: Input -> Output where var = 1 ...
;; ===============================================
;;
;; The subscripts can be described in the same way as Subscript DSL.
;; Note that -> Output is MUST and can't be omitted without macro: implict-einsum
;;
;;
;; Implict-Mode:
;; The shape of output MUST be described in the subscript.
;;
;;  A[~ i j] -> [~]
;;  A[~ i j] -> [scal] where scal = 1
;;

;;
;; Explict-mode:
;;  A[i i] -> B[i]
;;

;;
;; Broadcasting:
;; ~ represents broadcastable axes, that is, ~ is used in implict-mode, the returned tensor's corresponding axes are keeping broadcastable state.
;; (einsum A[i ~ j] -> [~]) adds a broadcastable axis for example.
;; => i <1 x N> j Tensor is returned.
;;



;;
;; Defactorizing matmul
;; When the number of args >= 3:
;; A[a b] @ B[c d] @ C[e f]
;;    ^ 一度に計算できないから
;; (A @ B) @ Cとかにする 計算量は頑張って最小化
;;


;;
;; Some combinations of einsum (e.g.: matmul, element-wise axpy, transpose) can be accelerated by OpenBLAS/CUDA Backend if the currently using device supports:
;; We replace such combination as possible as we can
;;
;; Note that cl-waffe2 is NOT EINSUM FIRST LIBRARY, but VIEW API FIRST LIBRARY.
;;

;; (einsum A[i j] B[j k] -> [i k]) matmul for example

;; setf ... A+=B
;;
;; Corresponds:
;;
;; (dotimes (i 100)
;;   (dotimes (j 100)
;;      (dotimes (k 100)                            |
;;         (setf (aref out i k)                     | replaced with axpy/view
;;               (* (aref x i j) (aref y j k))))))  | axpy((* x_ij y_jk), ->(out i k)) where ->=view
;;
;; ↓
;; dotimes i ... 100, j ... 100
;;   setf(->(out, 0, 0~100), ->(x, 0, 0), ->(y, 0, 0~100))
;;                        ...
;;   setf(->(out, i, 0~100), (* ->(x, i, j), ->(y, j, 0~100)))
;;

;; (einsum A[i i] -> B[i]) corresponds with:

;;
;; loop i = 0...n do
;;    (setf (aref B i) (+ (aref A i i)))
;;

;; i kに関してはdot productを取らない
;; 


;; 方針: EinsumのSubscriptからloop i = 0...nのデータ構造を作る
;; defpathでBLAS APIで置き換える条件を作るか、cl-waffe2のAPIを呼び出す
;; ADするには, そのままChainを繋げてもいいけど, なるべく何らかのAPIで置き換える
;; Shapeは確定している。symbolはsymbolのままに損失関数作ってそれを最小化するように上にデータ構造をSortする (計算量どうでもいいや)

;;(defstruct Compiled-Einsum)

;; Einsum has a three behaviour depending on the length of shape-before

;; 1. Dispatching Permute/View/Sum/Flexible
;; (einsum A[i j] -> [scal] where scal = 1) to summarize
;; (einsum A[i i] -> [i])   to diagnoal
;; (einsum A[i j] -> [i i]) to sum axis=1
;; (einsum A[a ~ b] -> [~]) to make make A broadcastable
;; (dispatched through routers) if no hit, expands the kernel.

;; 2. Normal einsum

;; 3. Defactorize inputs and find the minimum combination, calling 2.
#|
(defun compile-einsum (from to shape-before shape-after let-bindings)
  (cond
    ((= (length shape-before) 1)
     (make-transform
      (symbol-eq (car from) (car to))
      (car from)
      (car to)
      (car shape-before)
      (car shape-after)
      (map 'list #'car let-bindings)))
    
    ((= (length shape-before) 2)
     (let ((einsum (make-einsum from to shape-before shape-after (map 'list #'car let-bindings))))

       einsum))
    (T

     nil)))

|#
