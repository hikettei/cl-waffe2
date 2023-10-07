
(in-package :cl-waffe2-simd)

(defun try-loading-simd-extension (&key
				     (pathname (asdf:system-relative-pathname "cl-waffe2" "./cl-waffe2-simd/kernels/cl-waffe2-simd.so")))
  (handler-case
      (progn
	(load-foreign-library pathname)
	t)
    (error (c)
      (declare (ignore c))
      (warn "cl-waffe2 SIMD extension for the CPUTensor backend is not available now because cl-waffe2 could not find the cl-waffe2-simd.so shared library.~%Please compile it with `make build_simd_extension` and try it again.")
      nil)))

