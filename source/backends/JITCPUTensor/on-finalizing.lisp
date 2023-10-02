
(in-package :cl-waffe2/backends.jit.cpu)

(defparameter *lazy-c-source* nil)
(defparameter *compiling-ntime-count* 0)

(defparameter *known-functions* (make-hash-table :test #'eq))

;; [TODO] element-wise op fusion
;; [TODO] Use SLEEF Backend For Mathematical Kernel
;; [TODO] AVXnnn Intrinsics?
;; [TODO] Adjustable Shape Support View Offsets <- Include in tests
;; [TODO] Add: JITLispTensor

(defmethod on-finalizing-compiling ((device-name (eql 'JITCPUTensor)) iseq-fw iseq-bw)
  (let* ((jit-nodes (loop for inst in `(,@iseq-fw ,@iseq-bw)
			  if (typep (wfop-self inst) 'CPUJIT-Blueprint)
			    collect inst)))
    (when (and (not (string= *lazy-c-source* ""))
	       ;; wfop-op is created by doing (compile nil) or search from LUT
	       ;; This method records all functions of (wfop-op x)
	       ;; If this method encounter an unknown method, it indicates the function isn't also compiled by gcc.
	       (some #'(lambda (x) (null (gethash (wfop-op x) *known-functions*))) jit-nodes))
      (mapc
       #'(lambda (x)
	   (setf (gethash (wfop-op x) *known-functions*) T))
       jit-nodes)
      (load-foreign-function *lazy-c-source*))

    (setf *lazy-c-source* "")
    (values iseq-fw iseq-bw)))

