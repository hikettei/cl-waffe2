
(in-package :cl-user)

;; Abstract Testing Tool

(defpackage :cl-waffe2/tester
  (:use :cl :rove)
  (:export
   #:running-test
   ))

(in-package :cl-waffe2/tester)

(defun running-test ()
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




