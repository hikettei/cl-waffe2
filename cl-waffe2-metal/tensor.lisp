
(in-package :cl-waffe2/backends.metal)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun metal-available-p ()
    "Judges whether the computer supports Metal Graphics or not depending on machine-version."
    (and
     (string=
      "Apple"
      (subseq
       (machine-version)
       0 5))
     (string=
      "ARM64"
      (machine-type))))

  (unless (find :metal *features*)
    (when (metal-available-p)
      (push :metal *features*)))

  (declaim (inline metal-reject-p))
  (defun metal-reject-p ()
    #+metal(progn t)
    #-metal(progn nil)))

;; memo
;; CPUTensor MetalTensor interop is possible or not?
;; LispTensor -> CPUTensor?
(defclass MetalTensor (cl-waffe2/backends.lisp:LispTensor)
  nil
  (:documentation
   "## [AbstractTensor] MetalTensor
Provides Metal-Accelerated Operations
"))

(defmethod wf/t:current-backend-state ((backend-name (eql 'MetalTensor)))
  #+metal(format nil "Available (~a)" (machine-version))
  #-metal(format nil "Not Available"))

