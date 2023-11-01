
(in-package :cl-waffe2/vm.iterator)

;; Range is designed to replace view.

;; Range = (LIST FROM TO STEP <<LazyAxis>>)

;; TODO
;;    (Range 0 10 2)
;; +) (Range 2 8  2) ;; Applying further slicing for example
;; -------------------
;;    (Range 2 8  2)


(defstruct (Range
	    (:constructor range (from to &optional (step 1))))
  (from from :type fixnum)
  (to   to   :type fixnum)
  (step step :type fixnum))

