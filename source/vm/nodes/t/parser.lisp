
(in-package :cl-waffe2/vm.nodes.test)


(in-suite :test-nodes)

(defun test-bnf (subscript
		 excepted1
		 excepted2
		 excepted3)
  (multiple-value-bind
	(x y z)
      (cl-waffe2/vm.nodes::parse-subscript subscript)
    (and (equal x excepted1)
	 (equal y excepted2)
	 (equal z excepted3))))

(test bnf-parse-test
      (is (test-bnf `([x y] -> [z])
		    `((X Y))
		    `((Z))
		    NIL))
      (is (test-bnf `([x y z] [x y]->[z])
		    `((X Y Z) (X Y))
		    `((Z))
		    NIL)))

