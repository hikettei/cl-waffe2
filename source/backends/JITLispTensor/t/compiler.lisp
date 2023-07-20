
(in-package :cl-waffe2/backends.jit.lisp.test)

(defun test-case-tmp ()
  (with-no-grad
    (with-devices (JITLispTensor cl-waffe2/backends.lisp:LispTensor)
      (let ((a (!cos (!sin (forward (AddNode :float)
				    (randn `(10 10))
				    (randn `(10 10)))
			   :-> (!copy (randn `(10 10))))
		     :-> (randn `(10 10)))))
	(build a)))))

(defun test-case-tmp1 ()
  (with-no-grad
    (with-devices (JITLispTensor cl-waffe2/backends.lisp:LispTensor)
      (let ((a (forward (AddNode :float)
			(randn `(10 10))
			(randn `(10 10)))))
	(build a)))))

(in-suite :jit-lisp-test)

;; Check compiler can detect the change of shape, devices.
(test delimiting-compilable-nodes
  (is (let ((cl-waffe2/backends.jit.lisp::*compiling-ntime-count* 0))
	(test-case-tmp)
	(= cl-waffe2/backends.jit.lisp::*compiling-ntime-count* 2)))
  (is (let ((cl-waffe2/backends.jit.lisp::*compiling-ntime-count* 0))
	(test-case-tmp1)
	(= cl-waffe2/backends.jit.lisp::*compiling-ntime-count* 1))))



;; Test Case
;; A+=B
;; !add
;; (!copy (!sin (!copy A))) Sandwitch by Non-JITCompilable-Nodes

