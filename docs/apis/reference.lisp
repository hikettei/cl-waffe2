
(in-package :cl-waffe2.docs)

;; node to apis ha wakeru?


(with-page *base-impl-nodes* "Standard Nodes"
  (macrolet ((nodedoc (name)
	       `(insert "~a" ,(documentation (find-class name) 't)))
	     (caller-doc (name)
	       `(insert "~a" ,(documentation name 'function))))

    (nodedoc AddNode)
    (nodedoc SubNode)
    (nodedoc MulNode)
    (nodedoc DivNode)
    
    (nodedoc InverseTensorNode)
    
    (nodedoc ScalarAdd)
    (nodedoc ScalarSub)
    (nodedoc ScalarMul)
    (nodedoc ScalarDiv)

    (nodedoc MoveTensorNode)

    (nodedoc AbsNode)
    (nodedoc Scalar-AbsNode)

    (nodedoc SignNode)
    (nodedoc Scalar-SignNode)
    
    (nodedoc SqrtNode)
    (nodedoc Scalar-SqrtNode)

    (nodedoc SquareNode)
    (nodedoc Scalar-SquareNode)

    (nodedoc SinNode)
    (nodedoc Scalar-SinNode)

    (nodedoc CosNode)
    (nodedoc Scalar-CosNode)

    (nodedoc TanNode)
    (nodedoc Scalar-TanNode)

    (nodedoc ASinNode)
    (nodedoc Scalar-ASinNode)

    (nodedoc ACosNode)
    (nodedoc Scalar-ACosNode)

    (nodedoc ATanNode)
    (nodedoc Scalar-ATanNode)

    (nodedoc SinHNode)
    (nodedoc Scalar-SinHNode)

    (nodedoc CosHNode)
    (nodedoc Scalar-CosHNode)

    (nodedoc TanHNode)
    (nodedoc Scalar-TanHNode)

    (nodedoc ASinHNode)
    (nodedoc Scalar-ASinHNode)

    (nodedoc ACosHNode)
    (nodedoc Scalar-ACosHNode)

    (nodedoc ATanHNode)
    (nodedoc Scalar-ATanHNode)

    (nodedoc ExpNode)
    (nodedoc Scalar-ExpNode)

    (nodedoc Log2Node)
    (nodedoc Scalar-Log2Node)

    (nodedoc Log10Node)
    (nodedoc Scalar-Log10Node)

    (nodedoc LogENode)
    (nodedoc Scalar-LogENode)

    ;; TODO: pow/expt

    (nodedoc cl-waffe2/base-impl::LazyTransposeNode)

    (nodedoc ArgMax-Node)
    (nodedoc ArgMin-Node)

    (nodedoc MatmulNode)
    ))


(with-page *base-impl* "Basic APIs"
  (macrolet ((nodedoc (name)
	       `(insert "~a" ,(documentation (find-class name) 't)))
	     (caller-doc (name)
	       `(insert "~a" ,(documentation name 'function))))

    (caller-doc !matrix-add)
    (caller-doc !matrix-sub)
    (caller-doc !matrix-mul)
    (caller-doc !matrix-div)
    
    (caller-doc !inverse)
    (caller-doc !scalar-add)
    (caller-doc !scalar-sub)
    (caller-doc !scalar-mul)
    (caller-doc !scalar-div)
    
    (caller-doc !sas-add)
    (caller-doc !sas-sub)
    (caller-doc !sas-mul)
    (caller-doc !sas-div)
    
    (caller-doc !add)
    (caller-doc !sub)
    (caller-doc !mul)
    (caller-doc !div)

    (caller-doc !move)
    (caller-doc !copy)
    (caller-doc !copy-force)

    (caller-doc !reshape)
    (caller-doc !view)
    ;; unsqueeze/squeeze
    (caller-doc !flatten)
    (caller-doc !rankup)
    (caller-doc ->scal)
    (caller-doc ->mat)

    (caller-doc proceed)
    (caller-doc proceed-time)
    (caller-doc proceed-backward)

    (caller-doc !flexible)

    (caller-doc !abs)
    (caller-doc !sign)
    (caller-doc !sqrt)
    (caller-doc !square)
    (caller-doc !sin)
    (caller-doc !cos)
    (caller-doc !tan)
    (caller-doc !asin)
    (caller-doc !acos)
    (caller-doc !atan)
    (caller-doc !sinh)
    (caller-doc !cosh)
    (caller-doc !tanh)
    (caller-doc !asinh)
    (caller-doc !acosh)
    (caller-doc !atanh)

    (caller-doc !exp)
    (caller-doc !log2)
    (caller-doc !log10)
    (caller-doc !log)

    (caller-doc !sum)
    (caller-doc !mean)

    (caller-doc !argmax)
    (caller-doc !argmin)

    (caller-doc !t)
    (caller-doc !matmul)
    (caller-doc !dot)
    ))
