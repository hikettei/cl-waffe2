
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

    ))
