
(in-package :cl-waffe2/vm.iterator)

(defpackage :cl-waffe2/vm.iterator.test
  (:use
   :cl
   :fiveam
   :cl-waffe2/vm.generic-tensor
   :cl-waffe2/vm.iterator))

(in-package :cl-waffe2/vm.iterator.test)

(def-suite :iterator-test)
(in-suite  :iterator-test)

;; Expected to be working as well as for dynamic shapes
(test range-size-computation
  (is (= 10 (range-size
	     (range 0 10 1))))
  (is (= 5  (range-size
	     (range 0 10 2))))
  (is (= 5  (range-size
	     (range 0 10 -2)))))

;; Range [A, B)
(defun dynamic-do-range-test ()
  (wf/t:with-adjustable-symbol-scope
    (wf/t:with-adjustable-symbols ((A 10))
      (let ((count nil))
	(do-range (var (range 0 'A 1))
	  (push var count))
	(equal count
	       (loop for i downfrom 9 to 0 collect i))))))

;; Range [A, B)
(defun dynamic-do-range-test1 ()
  (wf/t:with-adjustable-symbol-scope
    (wf/t:with-adjustable-symbols ((A 8) (B 1))
      (let ((count nil))
	(do-range (var (range 'B 'A 1))
	  (push var count))
	(equal count
	       (loop for i downfrom 7 to 1 collect i))))))

(defun all-dynamic-do-range-test ()
  (wf/t:with-adjustable-symbol-scope
    (wf/t:with-adjustable-symbols ((A 2) (B 10) (C -2))
      (let ((count nil))
	(do-range (var (range 'A 'B 'C))
	  (push var count))
	(equal count
	       `(2 4 6 8 ))))))

(defun all-dynamic-do-range-test1 ()
  (wf/t:with-adjustable-symbol-scope
    (wf/t:with-adjustable-symbols ((A 2) (B 10) (C 2))
      (let ((count nil))
	(do-range (var (range 'A 'B 'C))
	  (push var count))
	(equal count
	       (reverse `(2 4 6 8)))))))

(defun rev-dynamic-do-range-test ()
  (wf/t:with-adjustable-symbol-scope
    (wf/t:with-adjustable-symbols ((A 2) (B 10) (C 2))
      (let ((count nil))
	(do-range (var (range 'B 'A 'C))
	  (push var count))
	(equal count `(2 4 6 8))))))

(test dynamic-do-range-test
  (is (dynamic-do-range-test))
  (is (dynamic-do-range-test1))
  (is (all-dynamic-do-range-test))
  (is (all-dynamic-do-range-test1))
  (is (rev-dynamic-do-range-test)))

(defun fixnum-do-range-test ()
  (let ((count nil))
    (do-range (var (range 0 10 1))
      (push var count))
    (equal count
	   (loop for i downfrom 9 to 0 collect i))))

;; Stepby=2
(defun range-test-1 ()
  (let ((count nil))
    (do-range (var (range 0 10 2))
      (push var count))
    (equal count
	   (loop for i from 8 downto 0 by 2 collect i))))

(defun range-test-2 ()
  (let ((count nil))
    (do-range (var (range 0 10 -1))
      (push var count))
    (equal count
	   (loop for i upfrom 0 below 10 collect i))))

(defun range-test-3 ()
  (let ((count nil))
    (do-range (var (range 2 10 -2))
      (push var count))
    (equal count
	   (loop for i upfrom 2 below 10 by 2 collect i))))

(test fixed-do-range-test
  (is (fixnum-do-range-test))
  (is (range-test-1))
  (is (range-test-2))
  (is (range-test-3)))

(defun compose-test ()
  (let ((a1 (.range
	     (range 2 6 2)
	     (range 2 8 1))))
    (and
     (= (range-from a1) 4)
     (= (range-to   a1) 8)
     (= (range-step a1) 2))))

(test range-compose-test
  (is (compose-test)))

(test range-compose-complicated-test
  (is (progn
	(let ((result (.range (range 4)
			      (range 0 5 -1))))
	  (and (= (range-step result) 1)
	       (= (range-from result) 0)
	       (= (range-to   result) 1)))))
  (is (progn
	(let ((result (.range (range 0 5)
			      (range 0 5 -1))))
	  (and (= (range-step result) 1)
	       (= (range-from result) 0)
	       (= (range-to   result) 5)))))
  (is (progn
	(let ((result (.range (range 0 5 -1)
			      (range 0 5))))
	  (and (= (range-step result) -1)
	       (= (range-from result) 0)
	       (= (range-to   result) 5))))))

(test range-nth-test
  (is (= 2 (range-nth (range 3 0 -1) 2)))
  (is (= 1 (range-nth (range 3 0 -1) 1)))
  (is (= 0 (range-nth (range 3 0 -1) 0)))

  (is (= 2 (range-nth (range 3 0 1) 0)))
  (is (= 1 (range-nth (range 3 0 1) 1)))
  (is (= 0 (range-nth (range 3 0 1) 2)))
  
  (is (= 2 (range-nth (range 0 3 1) 2)))
  (is (= 1 (range-nth (range 0 3 1) 1)))
  (is (= 0 (range-nth (range 0 3 1) 0)))

  (is (= 3 (range-nth (range 3 6 1) 0)))
  (is (= 4 (range-nth (range 3 6 1) 1)))
  (is (= 5 (range-nth (range 3 6 1) 2)))

  (is (= 5 (range-nth (range 3 6 -1) 0)))
  (is (= 4 (range-nth (range 3 6 -1) 1)))
  (is (= 3 (range-nth (range 3 6 -1) 2)))

  (is (= 5 (range-nth (range 6 3 1) 0)))
  (is (= 4 (range-nth (range 6 3 1) 1)))
  (is (= 3 (range-nth (range 6 3 1) 2))))

