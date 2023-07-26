
(in-package :cl-waffe2/vm.generic-tensor)

;;
;;
;;               [Function-name]
;;                  /        \
;;              [Device]  [Device]
;;              /     \   /      \
;;           [Dtype]   ...
;;           /
;;        [Stride]
;;          |
;; [Inlined Cached Function ID=0] [... ID=1...]
;;

;; 
;; Search-Route = Fname -> Device -> -> Dtype -> Tensor_Stride/Permute
;;


(defparameter *tensor-id-table* (make-hash-table)) ;; [Device] -> [Dtype] -> [Stride] -> [ID]

(declaim (inline gethash-with-id gethash-leaf lut-search-id))
(defun gethash-with-id (key table &key (test #'eq))
  (declare (type hash-table table)
	   (optimize (speed 3) (safety 0)))
  (apply #'values
	 (or (gethash key table)
	     (setf (gethash key table)
		   (list
		    (hash-table-count table)
		    (make-hash-table :test test))))))

(defun gethash-leaf (key table)
  (declare (type hash-table table))
  (or (gethash key table)
      (setf (gethash key table) (hash-table-count table))))

(defun lut-search-id (tensor)
  (declare (type AbstractTensor tensor)
	   (optimize (speed 3)))

  (multiple-value-bind (id1 dtype-tree)
      (gethash-with-id (class-name (class-of tensor)) *tensor-id-table*)
    (multiple-value-bind (id2 stride-tree)
	(gethash-with-id (dtype tensor) dtype-tree :test #'equal)
      (multiple-value-bind (id3 shape-tree)
	  (gethash-with-id (tensor-actual-stride tensor) stride-tree :test #'equal)
	;; (DEVICE DTYPE STRIDE SHAPE)
	`(,id1 ,id2 ,id3 ,(gethash-leaf (shape tensor) shape-tree))))))

;;(declaim (inline lut-search-function))
(defun lut-search-function (table function-name tensors &key (setme nil))
  (declare (type symbol function-name)
	   (type list tensors)
	   (type (or function null) setme)
	   (type hash-table table)
	   (optimize (speed 3)))
  ;; Function_Name + [Tensor_Ids] => Compiled Function

  (multiple-value-bind (id tensor-tree) (gethash-with-id function-name table :test #'equal)
    (declare (ignore id))
    (let ((keys (map 'list #'lut-search-id tensors)))
      (or (gethash keys tensor-tree)
	  (when setme
	    (setf (gethash keys tensor-tree) setme))))))




