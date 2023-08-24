
(in-package :gpt-2-example)

(defun load-npy (path &rest args)
  ;; npz -> AbstractTensor
  (format t "[INFO] load-npy attempts to load ~a...~%" (apply #'format nil path args))
  (parameter (change-facet (numpy-file-format:load-array (apply #'format nil path args)) :direction 'AbstractTensor)))

;; [TODO] Move this into :cl-waffe2/nn package
(defun gpt2-layernorm (x)

  )
