;;;; © 2023-2023 hikettei


;; Memo: English version is coming soon! :)
;; I want to write this document as neatly as possible, so please let me write it once in my first language, Japanese. T_T
;; But other documents are available on English: https://hikettei.github.io/cl-waffe2/

(in-package :cl-user)



;; ===============================================================================================================================
;; この文章には表記揺れがあります（ありました）
;;   1. Compositeとモデル Compositeはdefmodelで定義されるデータ構造です 他のライブラリでいうところのモデルに相当します。
;; ===============================================================================================================================


;; ===============================================================
;; cl-waffe2の公式ドキュメント:
;; https://hikettei.github.io/cl-waffe2/
;; 合わせてどうぞ：https://hikettei.github.io/cl-waffe2/overview/
;; ================================================================


;; ================================================================================================================
;; このファイルは直接読み込むというより、REPL上で動作されることを想定しています
;; もしSLIMEとEmacsでこのファイルを読んでいらっしゃるのでしたらC-c C-cで対応するコードを実行できます
;; 時間がある時に公式ドキュメントにも同様の内容を英訳して移植します (丁寧に書きたいので一旦日本語で下書きすることにしました)
;; ================================================================================================================









;; ====================
;;       導入
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

;;(asdf:load-system :cl-waffe2) <- これでもcl-waffe2を読み込めます

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
   :cl-waffe2/vm                ;; cl-waffe2のIRを動作させるための仮想マシン
   :cl-waffe2/vm.generic-tensor ;; AbstractTensorとJITコンパイラに関するAPIを提供
   :cl-waffe2/vm.nodes          ;; AbstractNode, つまり計算ノードを構築する時に使う拡張的なAPIを提供

   
   :cl-waffe2/backends.lisp     ;; cl-waffe2実装の一つ 全てCommon Lispで記述され、環境に依存しない。汎用性第一
   :cl-waffe2/backends.cpu      ;; cl-waffe2実装の一つ OpenBLAS等外部ライブラリを呼び出す。速度第一
   :cl-waffe2/backends.jit.cpu  ;; cl-waffe2 -> C/C++へのJITコンパイラ

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

;; 小さいスケールの行列やスカラー値に対しても高速に動作します:
(proceed-time (!matmul (randn `(1 1)) (randn `(1 1))))

;; 複数の計算を組み合わせてノードを構築していき、答えが欲しくなったタイミングでProceedを呼び出してください。
;; ~ Examples~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

;; 図を用いた解説など、詳しくはドキュメントをご覧ください：
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

;; Tips: disassemble-waffe2-ir関数を用いればcl-waffe2がどのように計算ノードをコンパイルしたのかを可視化できます
(disassemble-waffe2-ir (!sum (!softmax (parameter (randn `(100 100))))))

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

;; ~ 追記: whereについて ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; 計算ノードやモデルの定義など、ありとあらゆるところに:whereで宣言されてる謎の記法の言語は`Subscript DSL`と呼ばれるもので、静的型のないCommon Lispで演算前後の形状の状態や、またどのポインタをどう使うかといった情報を表しています。
;; 仕様については前述したドキュメント(https://hikettei.github.io/cl-waffe2/nodes/#introducing-subscript-dsl)を読んでください。
;;
;; cl-waffe2は至る所に:whereが張り巡らされています。(AbstractNodeはこれが必須, Compositeは任意になっています) これはコンパイルする前にネットワークの形状を特定するために必要になっていて、事前ShapingErrorやJIT, offsetを事前に求めておくといった主要な機能はこれが元になっています。
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
;; (i.e.: ユーザーが新しいデバイスを定義しても、その一部だけ再実装を与えれば十分）

;; 優先度:       高 <-  -> 低
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

;; Compositeはdefmodel-asマクロを介してNodeやFunctionに変換することができます。
;; トップレベルで呼び出せばコンパイルされた関数をキャッシュすることになります。

(defmodel-as (Softmax-Model) :asif :function :named !softmax-static)

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

;; このcall->とasnodeを組み合わせたのがdefsequenceで定義されるデータ構造です
;; defsequenceマクロは、defmodelを単にWrapしただけのマクロで、複数のノードやモデルを一列に並べるような構造を定義します。
;; もし定義するモデルが call-> と asnode を組み合わせるだけで実装できる場合、ぜひdefsequenceを使ってください。記述量が格段に減少します。

(defsequence MLP-Sequence (in-features hidden-dim out-features
			   &key (activation #'!relu))
	     "三層のMLPモデル" ;; <- 最初の引数は文字列にするとドキュメントに、空でもOK
	     (LinearLayer in-features hidden-dim)
	     (asnode activation)
	     (LinearLayer hidden-dim hidden-dim)
	     (asnode activation)
	     (LinearLayer hidden-dim out-features))

;; Printとするとよりモデルの構造がはっきりします：
(print (MLP-Sequence 784 512 256 :activation #'!relu))

#|
<Composite: MLP-SEQUENCE{W5800}(
    <<5 Layers Sequence>>

[1/5]          ↓ 
<Composite: LINEARLAYER{W5794}(
    <Input : ((~ BATCH-SIZE 784)) -> Output: ((~ BATCH-SIZE 512))>

    WEIGHTS -> (512 784)
    BIAS    -> (512)
)>
[2/5]          ↓ 
<Composite: ENCAPSULATED-NODE{W5792}(
    #<FUNCTION !RELU>
)>
[3/5]          ↓ 
<Composite: LINEARLAYER{W5786}(
    <Input : ((~ BATCH-SIZE 512)) -> Output: ((~ BATCH-SIZE 512))>

    WEIGHTS -> (512 512)
    BIAS    -> (512)
)>
[4/5]          ↓ 
<Composite: ENCAPSULATED-NODE{W5784}(
    #<FUNCTION !RELU>
)>
[5/5]          ↓ 
<Composite: LINEARLAYER{W5778}(
    <Input : ((~ BATCH-SIZE 512)) -> Output: ((~ BATCH-SIZE 256))>

    WEIGHTS -> (256 512)
    BIAS    -> (256)
)>)>
|#


;; ~~ [AbstractTensorの様相] ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; (内部実装の話なので興味なかったら読み飛ばしてください)
;; cl-waffe2のTensorは遅延評価をたくさん取り扱うので、NumpyやPyTorchのそれと異なる性質のものが欲しくなります。
;; cl-waffe2ではTensorの使用目的に合わせて、二種類の状態(:facet)を用意しました。(テンソルをPrintすると出てきますね）
;; :facetには:existと:inputの2種類があります（それぞれExistTensorとInputTensorという異なる性質を持った行列を表す）
;;
;; (make-input shape nil)関数は、メモリの割り当てを伴わない空の行列(InputTensor)を生成します。（つまりいくらでも生成し放題）
;; この行列に対して(tensor-vec tensor)関数でメモリにアクセスしようとすると初めて割り当てが行われます。

;; それに対して(make-tensor ...)や(randn ...)などの関数は、通常のメモリ割り当てを伴う行列(ExistTensor)を生成します。

;; InputTensorのその性質は、JITが計算ノードをトレスする実装や、計算の一時領域としてとりあえず作っておいて後から削除するみたいな動作と相性が良くて
;; cl-waffe2の至る所で用いられています。 ↑のENCAPSULATED-NODEは:WHEREで演算前後の遷移が定義されていないですが、ここでInputTensorが活躍してくれてうまくTraceします
;; 閑話休題
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;; ===========================
;;  ノード定義の豆知識など
;; ===========================

;; 1. define-static-node

;; defnodeとdefine-implはforwardを記述するのにマクロを書かないといけないし、defmodelは逆伝播を記述することができません。
;; :forwardと:backwardをdefunを書くのと同様の気持ちで記述したい場合はdefine-static-nodeを用います。

;; この場合, save-for-backwardはset-save-for-backward/read-save-for-backward関数を用いて手動で書くか、
;; with-setting-save4bw, with-reading-save4bwマクロを用いるかで、逆伝播の計算時に必要な計算ノードをコピーする必要があります。

;; Example: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(define-op (Static-Sin (self)
	    :where (A[~] -> OUT[~])
	    :save-for-backward-names (x-input)
	    :forward ((self x)
		      (print "Hi :) the operation sin is executed.")
		      (with-setting-save4bw ((x-input x))
			(proceed (!sin x))))
	    :backward ((self dout)
		       (with-reading-save4bw ((x x-input))
			 (values (proceed (!mul (!cos x) dout)))))))

;; 言い忘れたましたが、cl-waffe2のマクロで定義したデータ構造は
;; 全て定義名と同じ名前の関数がコンストラクタとして定義されています。
(print (Static-Sin)) ;; <Node: STATIC-SIN-T (A[~] -> OUT[~])>

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;; 遅延評価やマクロといった癖の強い機能を使わずにノードを定義できます
;;
;; ~ Tips ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  define-static-nodeはdefine-implとdefnodeを:device tで同時に定義するような実装になっています。
;; ここで、:device tでノードを実装すると他の全ての実装が無視され:device tのものしか採用されなくなります。
;; これは、ノードがもうすでに汎用的である場合だとか、ユーティリティ（PrintNode等）を実装する時に便利です。
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;; 2. reject-p

;; define-implで実装を提供しても、例えば一部のデータ型にだけはまだ対応していない・・・みたいな状況があります。
;; その場合 :reject-pに ノードを採用されたくない場合はTを返す関数を設定してください
;; 渡す関数: (コンスラクタの引数1 引数2 ...) -> Boolean
;; (P.S.: AddNode-Revisitノードのコンストラクタの第一引数がdtypeなのはこれのせいですね)

;; See also: https://hikettei.github.io/cl-waffe2/nodes/#tips-reject-p

;; 3. with-instant-kernel

;; 全てのcl-waffe2コードは、一旦Lispのプログラムに再コンパイルされてから実行されますが、
;; その際デバッグなどのために任意のCommon Lispを埋め込むことが可能です
;; with-instant-kernelマクロは、第一引数のTensorに続けて、CommonLispコードを埋め込むためのノードを呼び出し計算ノードを接続します：

(let* ((x (randn `(2 2)))
       (out (with-instant-kernel x
	      `(progn
		 (print ,x)
		 ,x))))
  (print (proceed out)))

#|
{CPUTENSOR[float] :shape (2 2)  
  ((1.6706548   1.0810136)
   (-0.40689966 0.7384378))
  :facet :exist
  :requires-grad NIL
  :backward NIL} 
{CPUTENSOR[float] :shape (2 2) :named ChainTMP5891 
  :vec-state [computed]
  ((1.6706548   1.0810136)
   (-0.40689966 0.7384378))
  :facet :input
  :requires-grad NIL
  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}
|#

;; lazy-print関数は上のPrintデバッグをもう少し合理的にするためのAPIで、

(let* ((x (parameter (randn `(10 10)))))
  (proceed-backward
   (!sum (lazy-print x))))

;; このような具合でForwardとBackwardの計算途中の状態を可視化します。
#|
===> [Forward] PrintNode: <Node: PRINTNODE-T (A[~] -> A[~])>  =========>
{CPUTENSOR[float] :shape (10 10) -> :view (<T> <T>) -> :visible-shape (10 10)  
  ((0.25651443   -0.40045857  -0.9287579   ~ 0.8480048    -0.40961027  2.3160353)                    
   (-1.6331433   1.1089627    0.16270746   ~ 0.50973755   -1.6394889   -0.047892045)   
                 ...
   (-0.39202097  1.1730671    0.32979876   ~ 1.1525645    -0.1317046   -0.9641139)
   (-0.5417676   0.1259635    -0.14945106  ~ -0.35147586  -0.029653043 -0.26303303))
  :facet :exist
  :requires-grad T
  :backward NIL}
<== [Backward] PrintNode: <Node: PRINTNODE-T (A[~] -> A[~])>  <========
Previous dout:
{CPUTENSOR[float] :shape (10 10) :named ChainTMP6008 
  ((1.0 1.0 1.0 ~ 1.0 1.0 1.0)           
   (1.0 1.0 1.0 ~ 1.0 1.0 1.0)   
        ...
   (1.0 1.0 1.0 ~ 1.0 1.0 1.0)
   (1.0 1.0 1.0 ~ 1.0 1.0 1.0))
  :facet :input
  :requires-grad NIL
  :backward NIL}
|#

;; 4. on-finalizing-compiling

;; このメソッドはcl-waffe2をさらに拡張してJITコンパイラを埋め込む時に使います: https://hikettei.github.io/cl-waffe2/nodes/#generic-on-finalizing-compiling
;; See also: my implementations of ./source/backends/JITLispTensor

;; ============================
;;  Useful APIS
;; ============================

;; 1. !view
;; 与えられた行列のViewを作成します。

;; ~ Examples ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;;                       
;;  (0 1 2)              (0 1 2)
;;  (3 4 5)  -> View ->  (+ + +)
;;  (6 7 8)              (+ + +)
;;

(print (!view (ax+b `(3 3) 1 0) `(0 1)))

;;{CPUTENSOR[float] :shape (3 3) -> :view (<(0 1)> <T>) -> :visible-shape (1 3) :named ChainTMP6130 
;;  :vec-state [maybe-not-computed]
;;  ((0.0 1.0 2.0))
;;  :facet :input
;;  :requires-grad NIL
;;  :backward <Node: VIEWTENSORNODE-T (A[RESULT] B[BEFORE] -> A[RESULT])>}

(print (!view (ax+b `(3 3) 1 0) `(1 2)))
;;{CPUTENSOR[float] :shape (3 3) -> :view (<(1 2)> <T>) -> :visible-shape (1 3) :named ChainTMP6138 
;;  :vec-state [maybe-not-computed]
;;  ((3.0 4.0 5.0))
;;  :facet :input
;;  :requires-grad NIL
;;  :backward <Node: VIEWTENSORNODE-T (A[RESULT] B[BEFORE] -> A[RESULT])>}


(print (!view (ax+b `(3 3) 1 0) 1 2))
;;{CPUTENSOR[float] :shape (3 3) -> :view (<1> <2>) -> :visible-shape (1 1) :named ChainTMP6210 
;;  :vec-state [maybe-not-computed]
;;  ((2.0))
;;  :facet :input
;;  :requires-grad NIL
;;:backward <Node: VIEWTENSORNODE-T (A[RESULT] B[BEFORE] -> A[RESULT])>}


;; 手動Broadcasting: 1 x 3行列を3 x 3行列かのようにRepeatする
(print (!view (ax+b `(1 3) 1 0) `(:broadcast 3)))


;;{CPUTENSOR[float] :shape (1 3) -> :view (<(BROADCAST 3)> <T>) -> :visible-shape (3 3) :named ChainTMP6216 
;;  :vec-state [maybe-not-computed]
;;  ((0.0 1.0 2.0)
;;   (0.0 1.0 2.0)
;;   (0.0 1.0 2.0))
;;  :facet :input
;;  :requires-grad NIL
;;  :backward <Node: VIEWTENSORNODE-T (A[RESULT] B[BEFORE] -> A[RESULT])>}


;; ~~~ Tips: Reshaper ~~~~~~~~~~~~~~~~~~
;; !permute !reshape !viewの三つの関数は、第一引数にラムダ関数を取ることができます。
;; その場合、第一引数以降の引数は無視され、第一引数のラムダ関数が返した値が、用いるSubscriptとなります。

;; 例1. 自動Broadcasting (broadcast-to tensor)は、tensorの形状と一致するようにBroadcastingする引数を返す関数です。

(let ((a (ax+b `(3 3) 0 0))
      (b (ax+b `(3 1) 1 0)))
  (print
   (proceed
    (!add
     a
     (!view b (broadcast-to a))))))

;;{CPUTENSOR[float] :shape (3 3) :named ChainTMP4505 
;;  :vec-state [computed]
;;  ((0.0 0.0 0.0)
;;   (1.0 1.0 1.0)
;;   (2.0 2.0 2.0))
;;  :facet :input
;;  :requires-grad NIL
;;  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}


;; 例2. reverse
;; 与えられたTensorのShapeを逆順にする場合はこのように記述します
;; 合成関数を作成するcomposeがあれば 単に(compose #'reverse #'shape)を渡せばOKです

(print
 (proceed
  (!reshape (ax+b `(5 3 2) 1 0)
	    #'(lambda (tensor)
		(reverse (shape tensor))))))

;;{CPUTENSOR[float] :shape (2 3 5) :named ChainTMP5019 
;;  :vec-state [computed]
;;  (((0.0  1.0  2.0  3.0  4.0)
;;    (5.0  6.0  7.0  8.0  9.0)
;;    (10.0 11.0 12.0 13.0 14.0))
;;   ((15.0 16.0 17.0 18.0 19.0)
;;    (20.0 21.0 22.0 23.0 24.0)
;;    (25.0 26.0 27.0 28.0 29.0)))
;;  :facet :input
;;  :requires-grad NIL
;;:backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}


;; 例3. torch-order
;; torch-orderはpermuteの引数の記法をNumpyやPyTorchと同じものにするReshaperです。(cl-waffe2のPermuteの記法はTorchと真逆です)

(print
 (proceed
  (!permute (ax+b `(3 5) 1 0) (torch-order 1 0)))) ;; (!permute ... 0 1)と同値

;;{CPUTENSOR[float] :shape (5 3) -> :view (<T> <T>) -> :visible-shape (5 3) :named;; ChainTMP5161 
;;  :vec-state [computed]
;;  ((0.0  5.0  10.0)            
;;   (1.0  6.0  11.0)   
;;         ...
;;   (3.0  8.0  13.0)
;;   (4.0  9.0  14.0))
;;  :facet :input
;;  :requires-grad NIL
;;:backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}

;; etc... (TODO: slice-step tflist indices)
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;; 2. !flexible
;; 行列のShapeの頭にBroadcast可能な次元を追加します

(print (!flexible (randn `(10))))
;;{CPUTENSOR[float] :shape (<1 x N> 10) :named ChainTMP6231 
;;  :vec-state [maybe-not-computed]
;;  (-0.50261    0.40090793  -0.19895083 ~ 0.20296319  -0.5383652  -1.4031166)
;;  :facet :input
;;  :requires-grad NIL
;;  :backward <Node: FLEXIBLE-RANK-NODE-T (A[~] -> A[~])>}

;; <1 x N>に対応する次元では、Broadcastingは自動で行われます
;; Broadcastingを用いた演算が行われると、この状態は解除されます

;; 10 x 10 Matrixと10 Vectorの加算
(print
 (proceed
  (!add
   (randn `(10 10))
   (!flexible (randn `(10))))))

;;{CPUTENSOR[float] :shape (10 10) :named ChainTMP6356 
;;  :vec-state [computed]
;;  ((-1.1468288   1.2965723    1.8768042    ~ 0.8179604    0.15577999   -1.8257952)                    
;;   (-1.9004688   -0.23534465  -0.3154031   ~ 0.32107505   1.0894032    2.2691782)   
;;                 ...
;;   (-0.5216994   -0.44841492  -0.5936316   ~ -0.5056487   0.6112499    -0.37115526)
;;   (-0.41688314  -0.6408297   0.66321605   ~ 1.0859071    -0.4954967   0.64494854))
;;  :facet :input
;;  :requires-grad NIL
;;:backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}

;; 時間があったらもう少しいいSemanticがないか考えたいですね:(
;; cl-waffe2のBroadcastingのルールはNumpy Semanticと異なるものにしようと考えていて、手動で何らかの宣言がないとRepeatやRankupがされないようにしています。(Optional Broadcasting)

;; 3. !permute
;; permuteは行列のSubscriptが呼ばれる順番を入れ替えます
;; 例：行列の転置を求める
(print
 (proceed
  (->contiguous
   (!permute (ax+b `(3 3) 1 0) :~ 0 1))))

;;{CPUTENSOR[float] :shape (3 3) :named ChainTMP6518 
;;  :vec-state [computed]
;;  ((0.0 3.0 6.0)
;;   (1.0 4.0 7.0)
;;   (2.0 5.0 8.0))
;;  :facet :input
;;  :requires-grad NIL
;;  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>} 

;; 4. !reshape
;; !reshapeは行列の形状を変化させます
;; tは引数で一度だけ使うことができ、その値は自動で推論されます

(print (proceed (!reshape (randn `(10 10)) t)))

;; 5. ->scal ->mat
;; ->scal ->mat関数は要素数が1の行列をスカラーに、スカラーを行列にするための関数です


(print (->scal (randn `(1 1))))

(print (->mat  (make-tensor 1.0)))

;; ==================================
;;  Optional Broadcasting
;; ==================================

;; cl-waffe2において、Broadcastingルールは自動で適用されません。

(!add (randn `(3 3)) (randn `(3 1))) ;; numpyでは1はブロードキャストされますが、cl-waffe2ではエラーになります。

;; numpy方式のbroadcastingは記述量が少なくなる反面、文脈によっては非自明な変換や意図しない動作を伴うことがあります。
;; そのため、cl-waffe2ではBroadcasting可能なShapeの範囲以外は自動でBroadcastingされないルールをしています。

;; 1. Broadcastingが呼び出される条件
;; 例えば加算ノードのwhere句: A[~] B[~] -> A[~]において~に対応する部分はBroadcast可能領域として宣言されます。~に対応する部分のShapeで, 次元数が一致しない 若しくは 対応する軸の値が一致しない場合, 次の関数に宣言された軸を追加していい領域に1を追加して!viewを通して:broadcastを適用し, もう一度加算ノードを呼び出します。

;; 1を追加してrepeatしていい領域は!flexible関数によって宣言されます。(%transformマクロを用いるとより可読性の高い方法で記述できます。)
;; 例えば

(print (!flexible (ax+b `(3 3) 1 0) :at 0))

;;{CPUTENSOR[float] :shape (<1 x N> 3 3) :named ChainTMP5932 
;;  :vec-state [maybe-not-computed]
;;  ((0.0 1.0 2.0)
;;   (3.0 4.0 5.0)
;;   (6.0 7.0 8.0))
;;  :facet :input
;;  :requires-grad NIL
;; :backward <Node: FLEXIBLE-RANK-NODE-T (A[~] -> A[~])>}

(print (!flexible (ax+b `(3 3) 1 0) :at 1))


;;{CPUTENSOR[float] :shape (3 <1 x N> 3) :named ChainTMP5932 
;;  :vec-state [maybe-not-computed]
;;  ((0.0 1.0 2.0)
;;   (3.0 4.0 5.0)
;;   (6.0 7.0 8.0))
;;  :facet :input
;;  :requires-grad NIL
;; :backward <Node: FLEXIBLE-RANK-NODE-T (A[~] -> A[~])>}

(print (!flexible (ax+b `(3 3) 1 0) :at 2))

;;{CPUTENSOR[float] :shape (3 3 <1 x N>) :named ChainTMP5932 
;;  :vec-state [maybe-not-computed]
;;  ((0.0 1.0 2.0)
;;   (3.0 4.0 5.0)
;;   (6.0 7.0 8.0))
;;  :facet :input
;;  :requires-grad NIL
;; :backward <Node: FLEXIBLE-RANK-NODE-T (A[~] -> A[~])>}

;; 新しく追加された<1 x N>という軸がまさにBroadcast可能な軸で、この軸はcl-waffe2が自由に1を追加したり、新しく追加した1をRepeatしていいことを宣言します。

;;
(print (proceed (!add (ax+b `(3 3) 0 0) (print (!flexible (ax+b `(3) 1 0) :at -1)))))

;;{CPUTENSOR[float] :shape (3 <1 x N>) :named ChainTMP5999 
;;  :vec-state [maybe-not-computed]
;;  (0.0 1.0 2.0)
;;  :facet :input
;;  :requires-grad NIL
;;  :backward <Node: FLEXIBLE-RANK-NODE-T (A[~] -> A[~])>} 
;;{CPUTENSOR[float] :shape (3 3) :named ChainTMP6070 
;;  :vec-state [computed]
;;  ((0.0 0.0 0.0)
;;   (1.0 1.0 1.0)
;;   (2.0 2.0 2.0))
;;  :facet :input
;;  :requires-grad NIL
;;:backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}

(print (proceed (!add (ax+b `(3 3) 0 0) (print (!flexible (ax+b `(3) 1 0) :at 0)))))

;;{CPUTENSOR[float] :shape (<1 x N> 3) :named ChainTMP6086 
;;  :vec-state [maybe-not-computed]
;;  (0.0 1.0 2.0)
;;  :facet :input
;;  :requires-grad NIL
;;  :backward <Node: FLEXIBLE-RANK-NODE-T (A[~] -> A[~])>} 
;;{CPUTENSOR[float] :shape (3 3) :named ChainTMP6153 
;;  :vec-state [computed]
;;  ((0.0 1.0 2.0)
;;   (0.0 1.0 2.0)
;;   (0.0 1.0 2.0))
;;  :facet :input
;;  :requires-grad NIL
;;  :backward <Node: PROCEEDNODE-T (A[~] -> A[~])>}

;; この例のように行/列ごとの加算なども自明に行うことができ、最小限の記述で意図しない動作を減らすことに役立ちます。

;; ==================================
;;  %transformマクロ
;; ==================================

;; %transformマクロはSubscript DSLと同じ文法で、!flexible !view !permuteを自明に表現するためのマクロです。

;; broadcast
;; 対応する~の位置に<1 x N>を追加します。

(let ((a (ax+b `(3 3) 1 0)))
  (print (%transform a[i j] -> [i ~ j])))

;;{CPUTENSOR[float] :shape (3 <1 x N> 3) :named ChainTMP6168 
;;  :vec-state [maybe-not-computed]
;;  ((0.0 1.0 2.0)
;;   (3.0 4.0 5.0)
;;   (6.0 7.0 8.0))
;;  :facet :input
;;  :requires-grad NIL
;;  :backward <Node: FLEXIBLE-RANK-NODE-T (A[~] -> A[~])>}

;; Permute

(let ((a (ax+b `(3 3) 1 0)))
  (print (%transform a[i j] -> [j i])))

;;{CPUTENSOR[float] :shape (3 3) -> :view (<T> <T>) -> :visible-shape (3 3) :named ChainTMP6186 
;;  :vec-state [maybe-not-computed]
;;  ((0.0 3.0 6.0)
;;   (1.0 4.0 7.0)
;;   (2.0 5.0 8.0))
;;  :facet :input
;;  :requires-grad NIL
;;  :backward <Node: LAZYTRANSPOSENODE-T (A[~ I J] -> A[~ I J])>} 

;; View

;; *1 *aのように記述すると対応する軸をBroadcastします。
(let ((a (ax+b `(1 3) 1 0)))
  (print (%transform A[i j] -> [*3 t])))

;;{CPUTENSOR[float] :shape (1 3) -> :view (<(BROADCAST 3)> <T>) -> :visible-shape ;;(3 3) :named ChainTMP6202 
;;  :vec-state [maybe-not-computed]
;;  ((0.0 1.0 2.0)
;;   (0.0 1.0 2.0)
;;   (0.0 1.0 2.0))
;;  :facet :input
;;  :requires-grad NIL
;;  :backward <Node: VIEWTENSORNODE-T (A[RESULT] B[BEFORE] -> A[RESULT])>}

;; Sliceやシンボルも使えます。
(let ((a (ax+b `(3 5 2) 1 0)))
  (print (%transform a[i j k] -> [(0 m) t 1] where m = 3)))

#|
{CPUTENSOR[float] :shape (3 5 2) -> :view (<(0 3)> <T> <1>) -> :visible-shape (3 5 1) :named ChainTMP6556 
  :vec-state [maybe-not-computed]
  (((1.0)              
    (3.0)    
     ...
    (7.0)
    (9.0))
   ((11.0)              
    (13.0)    
      ...
    (17.0)
    (19.0))
   ((21.0)              
    (23.0)    
      ...
    (27.0)
    (29.0)))
  :facet :input
  :requires-grad NIL
:backward <Node: VIEWTENSORNODE-T (A[RESULT] B[BEFORE] -> A[RESULT])>}
|#

;; ====================================================
;; AbstractTensorをCommon Lisp標準配列として取り扱う
;; ====================================================

;; change-facet関数を介して、AbstractTensorをあたかもCommon Lisp配列として取り扱ったり、Common Lisp配列をAbstract Tensorとして取り扱うことができます。

;; 2次元配列 -> AbstractTensorに変更する
(print
 (change-facet #2A((1 2 3)
		   (4 5 6)
		   (7 8 9))
	       :direction 'AbstractTensor))

;;{CPUTENSOR[int32] :shape (3 3)  
;;  ((1 2 3)
;;   (4 5 6)
;;   (7 8 9))
;;  :facet :exist
;;  :requires-grad NIL
;;  :backward NIL}

;; 変更の前後でコピーは作成されません。CL標準配列やAbstractTensorに適用された変更はお互いに適用されます！

;; AbstractTensor -> Common Lisp配列に変更する

;; 'arrayをdirectionに指定することで、形状を保ったまま変更
(print
 (change-facet (randn `(3 3)) :direction 'array))

;;#2A((0.87897 1.5162643 1.6645936)
;;    (-0.7619477 0.4606205 -0.057311922)
;;    (-0.052466746 0.4479398 0.37993735)) 

;; 'simple-arrayをdirectionに指定することで、内部で扱われている順番のまま変更
(print
 (change-facet (randn `(3 3)) :direction 'simple-array))
;; #(0.098267384 -0.68239963 1.9422209 -0.3872902 0.97872823 -0.96922004 0.8024853
;;  -0.90643305 -1.4785866) 

;; もしこのような変換の組み合わせを他の行列に対しても追加したい場合は、(convert-tensor-facet (from to))というジェネリック関数にメソッドを追加してください。

;; 詳細：https://hikettei.github.io/cl-waffe2/utils/#generic-convert-tensor-facet

;; change-facet関数を用いて、例えばcl-waffe2の機能だけでは足りないものを実装するとき、余分なコピーを用いずにnumclやmgl-mat等他のライブラリを使うことができます。

;; with-facet及びwith-facetsマクロはchange-facet関数をベースに実装されたマクロです。

(let ((a (randn `(3 3))))
  ;; AbstractTensorをCommonLisp配列としてアクセスする
  (with-facet (a* (a :direction 'array))
    ;; 対角を0で埋める
    (setf (aref a* 0 0) 0.0)
    (setf (aref a* 1 1) 0.0)
    (setf (aref a* 2 2) 0.0)
    (print a*))

  ;; この変更は元のAbstractTensorと同期されます。
  (print a))

;;#2A((0.0 -0.49273053 -1.081793)
;;    (0.07072042 0.0 0.06495718)
;;    (1.6785827 -0.85893154 0.0)) 
;;{CPUTENSOR[float] :shape (3 3)  
;;  ((0.0         -0.49273053 -1.081793)
;;   (0.07072042  0.0         0.06495718)
;;   (1.6785827   -0.85893154 0.0))
;;  :facet :exist
;;  :requires-grad NIL
;;  :backward NIL}

;; 遅延評価が邪魔になる場合、例えば学習データの読み込みなど、Common Lisp標準配列を介した場合もあります。そういった場合はfacetを変えて行列を直接編集してください :).

;; ==========================
;;  更なるJIT
;; ==========================

;; :cl-waffe2/backends.jit.cpuパッケージが提供するJITCPUTensorは
;; 同一形状の四則演算及び数学関数で構築される計算ノードをまとめてLispのコードにJITし、不要なコピーを更に減らし、(TO BE: lparallelで並列化する)などの手法で、さらなる高速化のためのバックエンドを提供します。

;; 例えばCPUTensorバックエンドであれば：
;;
;; (!sin (!sin tensor))
;;
;; [イメージ]:
;; OUT1 = (loop-with-view i ... do (move! place tensor i)
;; OUT2 = (loop-with-view i ... do (!sin OUT1 i)
;; OUT3 = (loop-with-view i ... do (!sin OUT2 i)
;;
;; のように内部で実行されます。(move!や!sinはOpenBLASやSLEEFの関数だと思ってください)

;; これがJITCPUTensorバックエンドでは:

;;
;; (!sin (!sin tensor))
;; [イメージ]
;; OUT = (loop-with-view i ... do (sin (sin tensor i)))

;; のように動的にCコードが生成されます (FuseOPs)


;; 簡易的なベンチマーク: CPUTensor + LispTensorの場合
(defun target-op ()
  (let ((a (randn `(100 100)))
	(b (randn `(100 100)))
	(c (randn `(100 100))))
    (!sin (!cos (!sin (!cos (!add a (!sub b c))))))))

(with-devices (CPUTensor LispTensor)
  (with-no-grad
    (proceed-time (target-op) :compile-mode :fastest)))

;;Evaluation took:
;;  0.000 seconds of real time
;;  0.000314 seconds of total run time (0.000258 user, 0.000056 system)
;;  100.00% CPU
;;  720,772 processor cycles
;;  260,960 bytes consed
  
;;Proceed-Time: Without allocation time:
;;Evaluation took:
;;  0.000 seconds of real time
;;  0.000100 seconds of total run time (0.000100 user, 0.000000 system)
;;  100.00% CPU
;;  227,324 processor cycles
;;  0 bytes consed

;; JITCPUTensor
(with-cpu-jit ()
  (with-no-grad
    (let ((model (build (target-op))))
      ;; コンパイル時間を除外して計測するためにBuild関数を用います
      (time (forward model))
      (time (forward model)))))

;;Evaluation took:
;;  0.002 seconds of real time
;;  0.002096 seconds of total run time (0.000679 user, 0.001417 system)
;;  100.00% CPU
;;  4,864,946 processor cycles
;;  256,976 bytes consed


;;Evaluation took:
;;  0.000 seconds of real time
;;  0.000504 seconds of total run time (0.000487 user, 0.000017 system)
;;  100.00% CPU
;;  1,157,624 processor cycles
;;  0 bytes consed

;; (P.S.: 数学関数のSIMD化が正しくできてるかまだ確認してません まだ、大規模な行列に関して並列化の処理はまだ実装れていません。これらは今後追加します。)


;;
;; (enable-cpu-jit-toplevel :viz-compiled-code t)
;; をトップレベルで宣言するとJITコンパイルされたCのコードがPrintされます
;;

;; ちなみにJITLispTensorバックエンドは開発者が2~3時間ほど適当にLispのコードを書くだけで実装できました。実際300行程度のコードになってるはずです。
;; いつかチュートリアルに任意のデバイスのJITコンパイラを書く方法みたいなのも追記します :)

;; ===========================
;;  Trainerとモデル最適化
;; ===========================

;; 最後に適当なニューラルネットワークを作って締めにしましょう
;; (ここで集中力が途切れる)
;; (TODO)
;;
;; 一部サンプルコードはここに記述してます: ./examples/sin_wave_mlp.lisp, ./examples/mnist.lisp
;; 

;; ================
;;  まとめ
;; ================

;; cl-waffe2はそもそも最近やっと行列演算/自動微分のベースラインが構築できたくらいで、
;; JuliaやPythonがやってる高度な数値計算を行うにはまだ力不足な感じがします。
;; （それはそうで、今年の6月くらいにリポジトリ作ってその後ぼちぼちやってるので時間が足りない・・・）
;; それを踏まえた上でcl-waffe2の長所短所を述べるなら：

;;
;;  長所
;;
;; 1. Common Lispで開発できる
;; 2. コンパイル時に扱える情報がたくさんある -> 将来的にもっと最適化を強くすることができるかも
;; 3. 環境に依存しない
;; 4. Common Lispは長い間仕様が変化してこなかったから、10年前に書いたライブラリだろうとずっと動作する
;;    -> cl-waffe2はこれになりたい。 早いうちにDeep Learningで必要な演算の定義(defnode)を強固にして
;;       CUDA/Metal/AVX, はたまた新しいバックエンドが出た時にも長続きするプログラムを書けるようにするのは目標
;; 5. REPLで使われることが前提のAPI 遅延評価を簡単に扱う
;; 6. 拡張性第一
;;     -> CL標準配列との互換性 すなわち CFFIで外部ライブラリを超簡単に呼び出せる
;; 7. (これからは) Coaltonがある cl-waffe2のAPIをCoaltonでWrapしたい。
;;
;;  短所
;;
;; 1. コンパイル時間の無視できないオーバーヘッド コンパイルをする回数を減らすようにコードを書く必要がある。
;;    -> define-composite-functionで事前にコンパイル/複数回コンパイルが必要な部分をCacheしようと考えている
;; 2. コミュニティ/他のライブラリの少なさ
;;    -> Hugging Faceとかはもちろん使えない... CommonLispからPythonのライブラリを呼び出す手段はあるにしろ
;;       Numpyを基盤にしたあのライブラリ群を一度使うと手間がかかる感が否めない
;; 3. 信頼性がない/機能が不十分
;;
;; 最初からcl-waffe2は自動微分を中心に限られた用途に特化しようと考えています。（最初から汎用的なものを目指すなど破滅を招くわ）
;;
;; MIT Licenceで公開しているのは僕の本職の受験勉強とかの傍らちまちま更新していこうかなと考えているからで、
;; 釘を刺すようですが本当に僕の趣味としてやっているに過ぎず、これが実用的になるにはあと数年くらいかかると思ってください。

;; 今すぐCommon Lispで実用的な行列演算がしたい！という方はまずnumclやmgl-mat, petalispにnumericals等...
;; 他の方から広く支持を得ているライブラリがありますので、ぜひそちらもご覧ください :)
;;
;; 拙い文章でしたが最後まで目を通していただきありがとうございます。何かありましたら気軽に連絡してください。
;; 多分これからもっと追記します
;;

