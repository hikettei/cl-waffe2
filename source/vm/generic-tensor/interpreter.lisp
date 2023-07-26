
(in-package :cl-waffe2/vm.generic-tensor)

;; Workflows of cl-waffe2 compiling:
;;
;; == [Interperter Mode (for proceed)] ===========================
;;
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;
;; [Global Compiled-Function Cache] <- Store Inlined functions
;;  MM 3Dx3D View=T View=T ...  -> Lambda ...
;;  MM 2Dx3D View=T View=T ...  -> Lambda ...
;; Move 3Dx3D View=T View=T ... -> Lambda ...
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;; [JIT Compiler Global Cache] <- Storing JIT Compiled functions (specialized on element-wise ops)
;;  a*b+c            -> Lambda ...
;;  mean(x) + sum(x) -> Lambda...
;;
;;
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;
;; out = Sum(Matmul(Copy(A), Copy(B)))
;;                   |
;;  [In-place mutation, fusion is applied...]
;;                   |
;;                   V
;;    <Interpreter> or <Build>
;;
;; If [interpreter] is chosen, each time we call operations, we look up [Global Compiled-Function-Cache] and [JIT Compiler Global Cache] each time.
;;
;; If [build] is chosen, on the other hand, the looking up time is inlined.
;;
;;

(defun run-node! (toplevel
		  &key
		    (stop-me nil)
		    (called-with-vars nil))
  "
## [function] run-node!
"
  (declare (type AbstractTensor toplevel)
	   (type boolean stop-me)
	   (optimize (speed 3)))

  (when (or stop-me
	    (null (tensor-state toplevel)))
    (return-from run-node! toplevel))

  (when (detach-p toplevel)
    (setq stop-me t))
  
  
  (let* ((state (tensor-state toplevel))
	 (vars  (tensor-variables toplevel))
	 (node  (tensor-backward toplevel))
	 (compiled-fw (statecontainer-forward-out-form state)))

    (let ((next-states (map 'list #'(lambda (x) (run-node! x :stop-me stop-me :called-with-vars toplevel)) vars)))
      (register-variables vars)
      
      (let ((results (multiple-value-list (apply #'funcall-cached-function compiled-fw next-states))))
	(nth (tensor-out-n toplevel) results)))))

