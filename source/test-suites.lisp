
(in-package :cl-user)

;; Abstract Testing Tool

(defpackage :cl-waffe2/tester
  (:use :cl :rove)
  (:export
   #:running-test
   ))

(in-package :cl-waffe2/tester)


(defun str->backend (name)
  (let ((available-backends (map 'list #'class-name (alexandria:flatten (cl-waffe2:find-available-backends)))))
    (loop for candidate in available-backends
	  if (equalp (symbol-name candidate) name)
	    do (return-from str->backend candidate))
    (error "Unknown backend: ~a~%Available List: ~a" name available-backends)))

(defun running-test (&rest backends)
  (when (null backends)
    (error "running-test: Specify more than one backends."))
  ;; TODO: switch to use rove
  (apply #'cl-waffe2:set-devices-toplevel (map 'list #'str->backend backends))
  (cl-waffe2:show-backends)

  ;; Test all ops defined in base-impl
  (rove:run-suite :cl-waffe2/base-impl.test)
  
  ;;(fiveam:run! :iterator-test)
  ;;(fiveam:run! :test-nodes)

  ;;(fiveam:run! :test-tensor)
;;  (fiveam:run! :base-impl-test)
;;  (fiveam:run! :jit-lisp-test)
		    
  ;;(fiveam:run! :lisp-backend-test)
  ;;(fiveam:run! :test-backends-cpu)
  ;;(fiveam:run! :jit-cpu-test)
		    
  ;;(fiveam:run! :nn-test)
  ;;(fiveam:run! :vm-test)
  )




