
(in-package :cl-waffe2/vm.nodes.generic-tensor.test)

(in-suite :test-tensor)

;; TODO: ViewTest
;; View->View, All Case
(defun test-array (shape)
  (cl-waffe2/distributions:ax+b
   shape
   1
   0
   :dtype :uint32))

(defun sliced-well-p (tensor start end &optional (offset 0))
  (let ((s (mref tensor start))
	(e (mref tensor end)))
    (and (= (round s) (+ offset start))
	 (= (round e) (+ offset end)))))

(defun M= (tensor list &aux (flag t))
  (loop for i upfrom 0
	for l in list
	unless (= (mref tensor i) l)
	  do (setq flag nil))
  flag)
	

;; Testing in 1d, all projection of view->view

;; T->T, T->Index, T->Slice, T->Slice-By
(test view-t->index-1d
  (is (= (round (mref (view (test-array `(10)) 0) 0) 0)))
  (is (= (round (mref (view (test-array `(10)) 1) 0) 1))))

(test view-t->slice-1d
  (is (sliced-well-p
	(test-array `(10))
	0
	9))
  (is (sliced-well-p
       (view (test-array `(10)) `(0 3))
	0
	3))
  (is (sliced-well-p
       (view (test-array `(10)) `(1 3))
	0
	3
	1))
  (is (sliced-well-p
       (view (test-array `(10)) `(2 -1))
	0
	6
	2)))

(test view-t->slice-1d-by ;; To ADD: by = -1 -2...
  (is (M= (view (test-array `(10)) `(0 10 2))
	  `(0 2 4 6 8)))
  (is (M= (view (test-array `(10)) `(2 10 2))
	  `(4 6 8))))

(test view-index->t-1d
  (is (M= (view (view (test-array `(10)) 1) t) `(1)))
  (is (M= (view (view (test-array `(10)) 0) t) `(0))))

(test view-slice->slice-1d
  (is (M= (view (view (test-array `(10)) `(1 5)) `(1 3)) `(2 3))))

(test view-slice->index-1d
  (is (M= (view (view (test-array `(10)) `(1 5)) 1) `(2))))

;; Slice-by isn't supported currently.
;; (test view-slice-by->index-1d
;;   (is (M= (view (view (test-array `(10)) `(1 10 2)) 1) `(6))))



	  
  
;; After then, 2d 3d 4d test...
