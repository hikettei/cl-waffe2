
(in-package :cl-waffe2/vm.nodes.generic-tensor.test)

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
(deftest view-t->index-1d
  (ok (= (round (mref (view (test-array `(10)) 0) 0)) 0))
  (ok (= (round (mref (view (test-array `(10)) 1) 0)) 1)))

(deftest view-t->slice-1d
  (ok (sliced-well-p
       (test-array `(10))
       0
       9))
  (ok (sliced-well-p
       (view (test-array `(10)) `(0 3))
       0
       3))
  (ok (sliced-well-p
       (view (test-array `(10)) `(1 3))
       0
       3
       1))
  (ok (sliced-well-p
       (view (test-array `(10)) `(2 10))
       0
       6
       2)))

(deftest view-t->slice-1d-by
  (ok (M= (view (test-array `(10)) `(0 10 2))
	  `(0 2 4 6 8)))
  (ok (M= (view (test-array `(10)) `(2 -1 2))
	  `(2 4 6 8)))
  (ok (M= (view (test-array `(10)) `(2 10 -2))
	  `(8 6 4 2))))

(deftest view-index->t-1d
  (ok (M= (view (view (test-array `(10)) 1) 0) `(1)))
  (ok (M= (view (view (test-array `(10)) 0) 0) `(0))))

(deftest view-slice->slice-1d
  (ok (M= (view (view (test-array `(10)) `(1 5)) `(1 3)) `(2 3))))

(deftest view-slice->index-1d
  (ok (M= (view (view (test-array `(10)) `(1 5)) 1) `(2))))

;; Slice-by isn't supported currently.
;; (test view-slice-by->index-1d
;;   (is (M= (view (view (test-array `(10)) `(1 10 2)) 1) `(6))))



	  
  
;; After then, 2d 3d 4d test...
