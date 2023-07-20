
;; Memo: English version is coming soon! :)
;; I want to write this document as neatly as possible, so please let me write it once in my first language, Japanese. T_T
;; But other documents are available on English: https://hikettei.github.io/cl-waffe2/

(in-package :cl-user)

;; この文章には表記揺れがあります（ありました）
;;   1. Compositeとモデル Compositeはdefmodelで定義されるデータ構造です 他のライブラリでいうところのモデルに相当します。
;;
					
;;
;; cl-waffe2の公式ドキュメント:
;; https://hikettei.github.io/cl-waffe2/
;;


;; このファイルは直接読み込むというより、REPL上で動作されることを想定しています
;; もしSLIMEとEmacsでこのファイルを読んでいらっしゃるのでしたらC-c C-cで対応するコードを実行できます
;; 時間がある時に公式ドキュメントにも同様の内容を移植します


;; ====================
;;   導入
;; ====================

;; cl-waffe2へようこそ! 開発者のhiketteiです :)

;; cl-waffe2はANSI Common Lisp上で動作する自動微分機能付き行列演算ライブラリで, 最終的に深層学習フレームワークとして提供することを目標に個人で開発しています。
;; cl-waffe2を使う上で最も特徴的な部分は「拡張性」と「遅延評価」に集約されるでしょう。ですがこれらが登場するのは、cl-waffe2を応用的に使う場合のみで
;; 通常はNumpyやPyTorchのようなシンプルなAPIから使えるものだと思ってください。

;; このファイルでは、利用用途別に整理したサンプルコードと解説を提供します
;; cl-waffe2の使い方の雰囲気を掴むのにぜひご活用ください :)


;; バグを発見された場合や、改善点などをお持ちの方はぜひIssueで共有するか、私に直接連絡してほしい（Githubのプロフィールに連絡先があります）です。
;; **cl-waffe2は開発中で機能が不安定ですから、実務などで使うことは絶対にやめてください 趣味や研究的な目線から見ていただけるとありがたいです**

;; ===============================
;;  準備：OpenBLASの設定
;; ================================


;; (現在ここだけ SBCL依存になってます＞＜ すぐ直す)


;; BLASは、線形代数で用いる基本的な演算を共通化するために作成されたAPIで、OpenBLASはその実装の一つです。
;; BLASのAPIに準拠したライブラリであれば何を使っていただいても構いません。
;; cl-waffe2ではCPUの行列積を主に一部の実装を高速化のためにOpenBLASを用いています。


;; 例えばaptが使える環境のLinuxでしたら
;; $ sudo apt install libblas-dev
;; でOpenBLASをインストールしてから、cl-waffe2が見つけられる場所に設定を書いてください。
;; 例えば・・・

(defparameter cl-user::*cl-waffe-config*
  `((:libblas "libblas.so")))

;; を実行してください。他のOSや処理系でも CFFIがOpenBLASの共用ライブラリを見つけることができたら大丈夫です
;; 環境によって名前は異なりますが、~/.roswell/init.elや~/.sbclrcなどの初期化ファイルに記述しておけば、次回から実行の手間が省けます


;; ===============================
;;  プロジェクトを始める
;; ==============================

;; cl-waffe2は大量のパッケージを提供するので、機能によって名前空間を分けています 
;; 一度自分のパッケージを定義してから、欲しいものだけを:useしてください。
;;
;; パッケージの定義がめんどくさかったら、(in-package :cl-waffe2-repl)で全ての機能が使えるパッケージを提供しています。試す時はどうぞ！


(load "./cl-waffe2.asd") ;; <- 見つからなかったらM-x slime-cdでディレクトリを変えてください


(defpackage :example-project
  (:use
   :cl
   ;; -------------------------------------------------------------------------------------------------------------
   ;; パッケージ名              |        説明
   :cl-waffe2                   ;; ネットワーク記述のためのユーティリティなどを提供
   :cl-waffe2/base-impl         ;; !addや!reshapeのような、基本的なAPIを呼び出す関数(およびその汎用定義)を提供
   :cl-waffe2/distributions     ;; 正規分布など、高速に確率分布をサンプリングするAPIを提供
   :cl-waffe2/nn                ;; 回帰モデルや誤差関数など、深層学習で用いるAPIを提供（開発中）
   :cl-waffe2/optimizers        ;; 数理最適化に関する実装を提供（開発中）
   :cl-waffe2/vm.generic-tensor ;; AbstractTensorとJITコンパイラに関するAPIを提供
   :cl-waffe2/vm.nodes          ;; AbstractNode, つまり計算ノードを構築する時に使う拡張的なAPIを提供

   
   :cl-waffe2/backends.lisp     ;; cl-waffe2実装の一つ 全てCommon Lispで記述され、環境に依存しない。汎用性第一
   :cl-waffe2/backends.cpu      ;; cl-waffe2実装の一つ OpenBLAS等外部ライブラリを呼び出す。速度第一
   :cl-waffe2/backends.jit.lisp ;; cl-waffe2実装の一つ cl-waffe2プログラムをCommon LispプログラムにJITするサンプルと実装を提供 (まだバグが多い・・・) with-no-grad宣言かじゃないと動かないです。

   ))

(in-package :example-project)

;; ====================================
;;  基本: 遅延評価
;; ====================================

;; 「最適化されてるかわからないノードは実行しない」はcl-waffe2開発におけるモットーの一つです。
;; どういうことかというと：

(print (+ 1 1))    ;; 即時実行され, 2が帰ってくる

(print (!add 1 1)) ;; !addを実行しても, 「+」は実行しない

;; => Return
;;{SCALARTENSOR[int32]  :named ChainTMP647 
;;  :vec-state [maybe-not-computed] **<- まだ実行されてない！**
;;  <<Not-Embodied (1) Tensor>>
;;  :facet :input
;;  :requires-grad NIL
;;  :backward <Node: SCALARANDSCALARADD-SCALARTENSOR (A[SCAL] B[SCAL] -> A[SCAL]
;;                                                    WHERE SCAL = 1)>}

;; この計算ノードを実行するには、ある地点でコンパイルを走らせる必要があります
;; REPLで使うなら`proceed`が一番使いやすいでしょう。

(print
 (proceed (!add 1 1)))

;;{SCALARTENSOR[int32]  :named ChainTMP675 
;;  :vec-state [computed] **<-実行された**
;;    2
;;  :facet :input
;;  :requires-grad NIL
;;  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>} 

;; (proceed tensor)や(build out)関数を通して初めて計算を実行することができます。

;; 複数の計算を組み合わせてノードを構築していき、答えが欲しくなったタイミングでProceedを呼び出してください。
;; ~ Examples ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(print
 (proceed (!add (!mul 3 2) 1)))

(print
 (proceed (!add (!mul 3.0 2.0) 1.0)))


;; randn ... 正規分布からサンプリングした3x3行列を返す
(print
 (proceed (!add (randn `(3 3)) (randn `(3 3)))))

;; Proceedの後に計算を続けることもできます。
;; Proceedは微分可能です

(print
 (proceed (!add (randn `(3 3))
		(proceed (!add (randn `(3 3))
			       (randn `(3 3)))))))

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;; 答えが必要になるタイミングをcl-waffe2に教えることで、さまざまな最適化が可能になります。
;; (例：自動でIn-place演算に置き換える, Viewのオフセットを事前計算, JITコンパイル, メソッド割り当て時間を排除する...)

;; Common Lispは静的型のない言語ですが、遅延評価のおかげで小さい行列に対しても高速に動作します。（となると、cl-waffe2は行列演算特化のDSLに近いと思います）

;; 図解されてる解説など、詳しくはドキュメントをご覧ください：
;; https://hikettei.github.io/cl-waffe2/overview/#compiling-and-in-place-optimizing

;; ====================================
;;  基本: コンパイル | 偏微分
;; ====================================

;;
;; build関数を用いることで計算ノードの順伝播と逆伝播をまとめてコンパイルできます。
;; また、テンソルを作成するときに、キーワードに:requires-grad tをつけるか、(parameter tensor)で書こうことで、cl-waffe2はそのテンソルをパラメーターと認識します。
;; パラメーターとは、計算ノード内で勾配が必要になるテンソルのことで、(grad tensor)で勾配が取り出せます。

;; build関数はCompiled-Compositeというクラスを返します。
;; (forward compiled-composite)でコンパイルされた順伝播、(backward compiled-composite)で逆伝播を呼び出します。

;; ~~ Example ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(defun sample-model (a b)
  (!sum (!matmul a b)))

(let ((a (parameter (randn `(10 10))))
      (b (parameter (randn `(10 10)))))

  (let ((compiled-model (build (sample-model a b))))
    (print compiled-model)

    (print (forward  compiled-model))
    (print (backward compiled-model))

    (print (grad a))
    (print (grad b))
    ))

;;<Compiled-Composite
;;    forward:  #<FUNCTION (LAMBDA () :IN "/private/var/tmp/slimeoTEyzi.fasl") {5397C39B}>
;;    backward: #<FUNCTION (LAMBDA () :IN "/private/var/tmp/slimeoTEyzi.fasl") {5398006B}>
;;
;;+= [Tensors in the computation node] =======+
;;
;;Subscripts:
;;
;;
;;Variables:
;; NAMES |  SIZE | 
;;
;;
;; - The number of tmp variables : 18
;; - The number of parameters    : 2
;;+========================================+
;;> 

#|
{CPUTENSOR[float] :shape (1 1) -> :view (<(BROADCAST 1)> <(BROADCAST 1)>) -> :visible-shape (1 1) :named ChainTMP1367 
  ((54.7542))
  :facet :input
  :requires-grad NIL
  :backward NIL} 
T 
{CPUTENSOR[float] :shape (10 10)  
  ((-6.5216365  3.4612074   5.76349     ~ 1.1767163   -0.35014528 5.7660365)                   
   (-6.5216365  3.4612074   5.76349     ~ 1.1767163   -0.35014528 5.7660365)   
                ...
   (-6.5216365  3.4612079   5.76349     ~ 1.1767161   -0.35014528 5.7660365)
   (-6.5216365  3.4612079   5.76349     ~ 1.1767161   -0.35014528 5.7660365))
  :facet :exist
  :requires-grad NIL
  :backward NIL} 
{CPUTENSOR[float] :shape (10 10)  
  ((-0.6177631 -0.6177631 -0.6177631 ~ -0.6177631 -0.6177628 -0.6177628)                  
   (4.3150215  4.3150215  4.3150215  ~ 4.3150215  4.3150215  4.3150215)   
               ...
   (1.7019254  1.7019254  1.7019254  ~ 1.7019254  1.7019254  1.7019254)
   (-1.5338145 -1.5338145 -1.5338145 ~ -1.5338145 -1.5338145 -1.5338145))
  :facet :exist
  :requires-grad NIL
:backward NIL}
|#

;; 学習の途中で変更したいテンソルがあったら、まずはそこを(make-input)で作成されたテンソルで置き換えてください。
;; (make-input shape 変数名)
;; shapeはシンボル名を含むことができます。

;;
;; set-inputメソッドで変数名に埋め込んたテンソルに値を授けることができます。
;; その際、Shape内部のシンボルは自動で置き換えられます。
;;

  
(let ((a (make-input `(batch-size 10) :train-data))
      (b (parameter (randn `(10 10)))))

  (let ((compiled-model (build (sample-model a b))))
    (print compiled-model)
    
    (set-input compiled-model :train-data (randn `(10 10))) ;; BATCH-SIZE x 10行列 <- 10 x 10行列

    (print (forward compiled-model))

    (set-input compiled-model :train-data (randn `(100 10))) ;; もっと大きい行列でもう一度

    (print (forward compiled-model))))

#|
<Compiled-Composite
    forward:  #<FUNCTION (LAMBDA () :IN "/private/var/tmp/slimeAwwPfI.fasl") {5397766B}>
    backward: #<FUNCTION (LAMBDA () :IN "/private/var/tmp/slimeAwwPfI.fasl") {539AA33B}>

+= [Tensors in the computation node] =======+

Subscripts:
     [BATCH-SIZE -> ?, max=?]


Variables:
   NAMES    |       SIZE       | 
––––––––––––––––––––––––––––––––
 TRAIN-DATA |  (BATCH-SIZE 10) | 


 - The number of tmp variables : 18
 - The number of parameters    : 1
+========================================+
> 
{CPUTENSOR[float] :shape (1 1) -> :view (<(BROADCAST 1)> <(BROADCAST 1)>) -> :visible-shape (1 1) :named ChainTMP2099 
  ((38.85525))
  :facet :input
  :requires-grad NIL
  :backward NIL} 
{CPUTENSOR[float] :shape (1 1) -> :view (<(BROADCAST 1)> <(BROADCAST 1)>) -> :visible-shape (1 1) :named ChainTMP2099 
  ((-84.13232))
  :facet :input
  :requires-grad NIL
 :backward NIL}
|#

;; =====================================
;;  基本: 計算ノードの単位
;; =====================================

;; defnodeマクロは、計算ノードの抽象的な定義を宣言します。

(defnode (AddNode-Revisit (myself dtype)
            :where (A[~] B[~] -> A[~]) ;; 詳しくは: https://hikettei.github.io/cl-waffe2/nodes/#introducing-subscript-dsl
            :documentation "A <- A + B"
            :backward ((self dout dx dy)
                       (declare (ignore dx dy))
                       (values dout dout))))

;; 計算ノードの抽象的な単位, AbstractNodeは以下の情報を持ちます
;;
;; 1. 演算前後の形状の遷移 :whereで記述されています。
;; 2. (任意) 逆伝播の定義, save-for-backward宣言
;; ...

;; このような抽象的なノードに対して、そのノードの実装はユーザーや標準で定義されたデバイスごとに提供されます。
;; デバイスとは、AbstractTensorを継承した全てのクラスのことです。
;; MyTensorという新しいデバイスを作って、加算命令を実装してみましょう。今から実装するデバイスと同じタイプの行列に対して実装が既にある場合, そのクラスを継承するだけで十分です

;; MyTensor extends LispTensor extends AbstractTensor
(defclass MyTensor (LispTensor) nil)

;; with-devicesマクロを用いて用いるデバイスを切り替えることができます。
(with-devices (MyTensor)
  (print (ax+b `(10 10) 1 0)))

;;{MYTENSOR[float] :shape (10 10)  
;;  ((0.0  1.0  2.0  ~ 7.0  8.0  9.0)            
;;   (10.0 11.0 12.0 ~ 17.0 18.0 19.0)   
;;         ...
;;   (80.0 81.0 82.0 ~ 87.0 88.0 89.0)
;;   (90.0 91.0 92.0 ~ 97.0 98.0 99.0))
;;  :facet :exist
;;  :requires-grad NIL
;;  :backward NIL} 

;; define-implマクロを介してMyTensorに対するAddNode-Revisitの実装を与えることができます。

(define-impl (AddNode-Revisit :device MyTensor)
	     :forward ((self x y)
		       (declare (ignore y))
		       (print "This form is called only after AddNode-Revisit is created")
		       `(progn
			  (print "This form is called only after AddNode-Revist is compiled")
			  (print "Describe here how AddNode is computed.")
			  ;; 複雑なのでXをそのまま返すだけにします。
			  ,x)))

(with-devices (MyTensor)
  ;; ノードを初期化
  (print (AddNode-Revisit :float)) ;;<Node: ADDNODE-REVISIT-MYTENSOR (A[~] B[~] -> A[~])>

  (print (call (AddNode-Revisit :float) (randn `(3 3)) (randn `(3 3))))
  ;; "This form is called only after AddNode-Revisit is created"

  (with-no-grad
    (print (proceed (forward (AddNode-Revisit :float)
			     (randn `(3 3)) (randn `(3 3)))))
    ;; "This form is called only after AddNode-Revist is compiled"
    
;;  {MYTENSOR[float] :shape (3 3) :named ChainTMP3378 
;;     :vec-state [computed]
;;  ((1.6247343   -1.5651377  1.1313442)
;;   (-1.5211822  -0.39717743 0.16194573)
;;   (1.9562016   0.10735343  0.28459883))
;;  :facet :input
;;  :requires-grad NIL
;;  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>} 
    ))

;; MyTensorは他の命令の実装がありませんが、その場合はwith-devicesの優先順位を調整してください

;; 扱うポインタに互換性のあるものから代理としてLispTensorの実装を使い計算します
;; (i.e.: 新しいデバイスを実装しても、その一部だけ再実装を与えれば十分）

(with-devices (MyTensor LispTensor)
  (print (proceed (!mul (randn `(3 3)) (randn `(3 3))))))

;;{MYTENSOR[float] :shape (3 3) :named ChainTMP3550 
;;  :vec-state [computed]
;;  ((-0.09925148  -2.900542    -0.7412353)
;;   (-0.020097438 1.2127453    1.2844063)
;;   (-0.15744707  -0.24318622  -3.1791134))
;;  :facet :input
;;  :requires-grad NIL
;;  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>} 

;; =====================================
;;  基本: Composite
;; =====================================

;; AbstractNodeを複数組み合わせた計算ノードの単位がCompositeで、defmodelマクロで定義できます。

(defmodel (Softmax-Model (self)
	   :where (X[~] -> [~])
	   :on-call-> ((self x)
		       (declare (ignore self))
		       (let* ((x1 (!sub x (!mean x  :axis 1 :keepdims t)))
                              (z  (!sum   (!exp x1) :axis 1 :keepdims t)))
                         (!div (!exp x1) z)))))

;; 普通にcallでCompositeを呼び出すことも可能ですし
(print (proceed-time (call (Softmax-Model) (randn `(10 10)))))

;; define-composite-functionマクロで事前にコンパイルしておき
;; それ以降は静的に動作する関数を作ることもできます

(define-composite-function (Softmax-Model) !softmax-static)

(print (!softmax-static (randn `(10 10))))

(print (time (!softmax-static (randn `(10 10)))))

;; 
;; [Composite] <= Node1 + Node2 + Node3...
;;

;; 深層学習においてCompositeは、モデルに相当するでしょう。


;; ==================
;;  形状の事前検査
;; ==================

;; cl-waffe2は遅延評価ベースのライブラリですから、コンパイル前に行列の形状検査などのエラーを検知できます：

(!add (randn `(100 100)) (randn `(3 3)))

;; (時間がないのである情報を並べるだけのエラーになってますが、もっと洗練したいですね・・・）

;; ===============================
;;  ネットワークの形状事前特定
;; ===============================

;; cl-waffe2は特別な設定なしに、全ての計算ノードの期待する入出力の形状を特定することができます。

(let ((a (LinearLayer 30 10)))
  (print a)
  (call a (randn `(10 30)))
  nil)

;;<Composite: LINEARLAYER{W5282}(
;;    <Input : ((~ BATCH-SIZE 30)) -> Output: ((~ BATCH-SIZE 10))>

;;    WEIGHTS -> (10 30)
;;    BIAS    -> (10)
;;)> 

;; ============================
;;  複数のモデルを合成する
;; ============================

;; 例えばLinearして活性化して... を繰り返すネットワークは合成関数です
;; Common Lispでそれを書くとネストが深くなってしまいます
;; defsequenceマクロとcall->マクロを使うと簡潔に記述できて便利です：

(print
 (proceed

  (call-> (randn `(3 3))       ;; 第一引数に対して...
	  
	  (asnode #'!add 1.0)  ;;  |
	  (asnode #'!relu)     ;;  | この流れでノードを組んでいく
	  (asnode #'!sum))))   ;;  ↓

;;{CPUTENSOR[float] :shape (1 1) -> :view (<(BROADCAST 1)> <(BROADCAST 1)>) -> :visible-shape (1 1) :named ChainTMP5579 
;;  :vec-state [computed]
;;  ((6.3379784))
;;  :facet :input
;;  :requires-grad NIL
;;  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>} 


;; !addや!relu, !sumはcl-waffe2の関数で、正真正銘`defun`を介して定義されます
;; 通常これらを呼び出すなら`funcall`が適切ですが、CompositeやAbstractNodeを呼び出すためにはcl-waffe2の`call`を用いる必要があるので困ります。（いろんなデータ型が混じっちゃってコードが汚くなります）

;; そのような時のために(asnode function &rest more-inputs)マクロが提供されています。
;; これは簡単に言うと、第一引数で受け取った`function`関数をEncapsulated-NodeというモデルでWrapして、callできるようにするマクロです。
;; more-inputsを指定すると、対応する位置に引数を追加します(e.g.: (asnode #'!add 1.0) -> (!add previous-input 1.0)に対応)

(print (asnode #'!sum))

;;<Composite: ENCAPSULATED-NODE{W5583}(
;;    #<FUNCTION !SUM>
;;)> 

(print (asnode #'!add 1.0))

;;<Composite: ENCAPSULATED-NODE{W5585}(
;;    #<FUNCTION (LAMBDA (CL-WAFFE2::X)
;;                 :IN
;;                 "/Users/hikettei/Desktop/cl-waffe-workspace/progs/develop/cl-waffe2/examples/tutorial_jp.lisp") {5397553B}>
;;)>

(print (call (asnode #'!relu) (randn `(3 3))))

;;{CPUTENSOR[float] :shape (3 3) :named ChainTMP5770 
;;  :vec-state [maybe-not-computed]
;;  <<Not-Embodied (3 3) Tensor>>
;;  :facet :input
;;  :requires-grad NIL
;;  :backward <Node: MULNODE-LISPTENSOR (A[~] B[~] -> A[~])>} 
