
(in-package :cl-waffe2/benchmark)


(defun waffe2-profile (function)
  (sb-profile:profile
   "CL-WAFFE2/BACKENDS.JIT.CPU"
   "CL-WAFFE2/VM.GENERIC-TENSOR"
   "CL-WAFFE2/BASE-IMPL")

  (sb-ext:gc :full t)
  (funcall function)

  (sb-profile:report)
  (sb-profile:unprofile
   "CL-WAFFE2/BACKENDS.JIT.CPU"
   "CL-WAFFE2/VM.GENERIC-TENSOR"
   "CL-WAFFE2/BASE-IMPL"))

(defun profile-compiler ()

  (waffe2-profile
   #'(lambda ()
       (with-cpu-jit ()
	   (build (!add 1.0 1.0))))))
