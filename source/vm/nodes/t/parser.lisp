
(in-package :cl-waffe2/vm.nodes.test)

(in-suite :test-nodes)

;; Enhancements(TODO)
;; Support Type   e.g.: Int:[x y] Float:[x y] -> []
;; Support Scalar e.g.: Int, [x y] -> ...

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
  (is (test-bnf `([x y] [x y]-> [z])
		`((X Y) (X Y))
		`((Z))
		NIL))
  (is (test-bnf `([x y] [x y]-> [z] [x])
		`((X Y) (X Y))
		`((Z) (X))
		NIL))
  (is (test-bnf `([x y] [x y]-> [z] [x] where x = 1 y = (aref a 0))
		`((X Y) (X Y))
		`((Z) (X))
		`((X 1) (Y (AREF A 0)))))
  (is (test-bnf `([x y z] [x y]->[z])
		`((X Y Z) (X Y))
		`((Z))
		NIL)))

