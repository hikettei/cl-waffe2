
(in-package :cl-waffe2.docs)

(with-page *overview* "Overview"
  (with-section "cl-waffe2, programmable deep-learning framework."
    (image "https://github.com/hikettei/cl-waffe/blob/main/docs/cl-waffe-logo.png?raw=true")
    (insert "⚠️ cl-waffe2 is still in the concept stage. Things are subject to change.")

    (insert "cl-waffe2 provides a set of differentiable matrix operations which is aimed to apply to build neural network models on Common Lisp.")

    (insert "The primary concepts and goals of this project is following:")
    (with-enum
      (item "One cl-waffe2 code, consisted of multiple and user-extensible backends, which consisted of small pieces of backends.")
      (item "Everything is lazy-evaluated, being compiled.")
      (item "defined-and-run, closer to defined-by-run APIs."))

    (insert "...")
    ))

