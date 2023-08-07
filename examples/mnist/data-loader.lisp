
(in-package :mnist-example)

(defun load-npy (path)
  (change-facet (load-array path) :direction 'AbstractTensor))

(format t "[INFO] Loading ./data/train_data.npy...~%")
(defparameter *train-data* (load-npy "examples/mnist/data/train_data.npy"))

(format t "[INFO] Loading ./data/train_label.npy...~%")
(defparameter *train-label* (load-npy "examples/mnist/data/train_label.npy"))


(format t "[INFO] Loading ./data/test_data.npy...~%")
(defparameter *test-data* (load-npy "examples/mnist/data/test_data.npy"))

(format t "[INFO] Loading ./data/test_label.npy...~%")
(defparameter *test-label* (load-npy "examples/mnist/data/test_label.npy"))

