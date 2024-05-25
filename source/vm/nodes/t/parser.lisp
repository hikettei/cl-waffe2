
(in-package :cl-waffe2/vm.nodes.test)

;; Enhancements(TODO)
;; Support Type   e.g.: Int:[x y] Float:[x y] -> []
;; Support Scalar e.g.: Int, [x y] -> ...

(defun test-bnf (subscript
		 expected1
		 expected2
		 expected3)
  (multiple-value-bind
	(is bs x y z)
      (cl-waffe2/vm.nodes::parse-subscript subscript)
    (declare (ignore is bs))
    (and (equal x expected1)
	 (equal y expected2)
	 (equal z expected3))))

(deftest bnf-parse-test
  (ok (test-bnf `([x y] -> [z])
		`((X Y))
		`((Z))
		NIL))
  (ok (test-bnf `([x y] [x y]-> [z])
		`((X Y) (X Y))
		`((Z))
		NIL))
  (ok (test-bnf `([x y] [x y]-> [z] [x])
		`((X Y) (X Y))
		`((Z) (X))
		NIL))
  (ok (test-bnf `([x y] [x y]-> [z] [x] where x = 1 y = (aref a 0))
		`((X Y) (X Y))
		`((Z) (X))
		`((X 1) (Y (AREF A 0)))))
  (ok (test-bnf `([x y z] [x y]->[z])
		`((X Y Z) (X Y))
		`((Z))
		NIL)))

