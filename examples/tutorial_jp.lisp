
;; Memo: English version is coming soon! :)
;; I want to write this document as neatly as possible, so please let me write it once in my first language, Japanese. T_T
;; But other documents are available on English: https://hikettei.github.io/cl-waffe2/

(in-package :cl-user)

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

;; =========================
;; プロジェクトを始める
;; =========================


;; ...
