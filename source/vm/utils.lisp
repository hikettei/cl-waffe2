
(in-package :cl-waffe2/vm)

;; Reading List
;; https://www.cspp.cc.u-tokyo.ac.jp/hanawa/class/spc2016s/sp20160426.pdf
;; https://www.r-ccs.riken.jp/wp/wp-content/uploads/2020/09/katagiri190516.pdf

(defun compose (&rest fns)
  (if fns
      (let ((fn1 (car (last fns)))
            (fns (butlast fns)))
        #'(lambda (&rest args)
                   (reduce #'funcall fns
                           :from-end t
                           :initial-value (apply fn1 args))))
      #'identity))

(declaim (ftype (function (AbstractTensor) list) topological-sort))
#+sbcl(setf sb-ext:*inline-expansion-limit* 30)
(defun topological-sort (var)
  (declare (type AbstractTensor var)
	   (optimize (speed 3)))
  (let ((seen nil)
	(top-sort nil))
    (declare (type list seen top-sort))
    (labels ((top-sort-helper (v is-leaf-p)
	       (if (or (find (the symbol (tensor-iid v))
			     seen :key #'tensor-iid :test #'eql)
		       ;;(null (tensor-backward v))
		       is-leaf-p)
		   nil
		   (progn
		     (push v seen)
		     (dolist (prev (tensor-variables v))
		       (top-sort-helper prev (detach-p v)))
		     (push v top-sort)))))
      #+sbcl(declare (inline top-sort-helper))
      (top-sort-helper var (detach-p var))
      (reverse top-sort))))

;; Autograd:
(defun make-backward-wfinst (tensor dout-prev)
  (when (and (tensor-compiled-instruction-cache-bw tensor)
	     (equal (car (last (tensor-compiled-instruction-cache-bw tensor))) ;; == Variables
		    (tensor-variables tensor)))
    (let ((result-tmp (tensor-compiled-instruction-cache-bw tensor)))
      (return-from make-backward-wfinst (apply #'values result-tmp))))

  (multiple-value-bind (bw-kernel iseq out-to dir) (make-backward tensor dout-prev)
    (declare (type (or null function) bw-kernel))
    (when (null bw-kernel) (return-from make-backward-wfinst nil))

    (let ((result
	    (list
	     bw-kernel
	     #'(lambda ()
		 (format nil "Block -> ~a-BACKWARD {
~a    }
  "
			 (class-name (class-of (tensor-backward tensor)))
			 (with-output-to-string (out)
			   (with-indent-to iseq
			     (dolist (i iseq)
			       (let ((*node-indent* (+ 4 *node-indent*)))
				 (format out "        ~a" i)))))))
	     out-to
	     dir
	     iseq
	     ;; Variables ... To detect the change of network.
	     (tensor-variables tensor))))

      (setf (tensor-compiled-instruction-cache-bw tensor) result)
      (apply #'values result))))

(defun tensor-compiled-kernel (tensor)
  (when (tensor-state tensor)
    (statecontainer-forward-out-form (tensor-state tensor))))

(defparameter *node-indent* 4)

(defun broadcasted-p (tensor)
  (some #'zerop (cl-waffe2/vm.generic-tensor::tensor-actual-stride tensor)))

(defun node-out-to (node) (cl-waffe2/vm.nodes::node-out-to node))

(defun init-state-container! (tensor)
  (when (null (tensor-state tensor))
    (setf (tensor-state tensor)
	  (make-statecontainer :forward-out-form (make-compiled-kernel)))))

(defun setq-vm-wrap-f ()
  "To avoid iseq=null, adds this node"
  "Setq{%VMWrap}")

(defun %vm-wrap-tensor (tensor)
  (init-state-container! tensor)
  (make-wfop
   #'(lambda (x) (declare (ignore x)) tensor)
   tensor
   #'setq-vm-wrap-f
   (list tensor)
   :out-to (list tensor)))

(defun sv4bw-p (node)
  (and (movetensor-p node) ;; MoveTensor(SAVE_FOR_BACKWARD) isn't subject to backward. just move tensors
       (cl-waffe2/base-impl:mv-lazy-sv4bw node)))


(defun expand-gradient-adder (tensor grad)
  ;; Tensor += Grad
  (setf (detach-p grad) t)
  (let ((out
	  (prog1
	      (let ((*no-grad* t))
		(reverse
		 (if (scalar-p tensor)
		     (progn
		       (node-compile-into-vm
			(forward
			 (cl-waffe2/base-impl::ScalarAndScalarAdd)
			 (grad tensor)
			 grad)))
		     (if (= (tensor-grad-count tensor) 0)
			 (progn
			   (incf (tensor-grad-count tensor) 1)
			   (node-compile-into-vm
			    (forward
			     (cl-waffe2/base-impl:MoveTensorNode
			      (dtype tensor)
			      :save-for-backward
			      t)
			     (grad tensor)
			     grad)))			 
			 (progn
			   (node-compile-into-vm
			    (forward
			     (cl-waffe2/base-impl:AddNode (dtype tensor))
			     (grad tensor)
			     grad)))))))
	    (setf (detach-p grad) nil))))
    (setf (wfop-grad-adder-p (car out)) t)
    out))

(defun render-debug-info ()
  "Displays all global variables related to the error"
  (with-output-to-string (out)
    (format out "= [DebugInfo] =====~%")
    (format out "Global Variables:~%")
    (format out "Dynamic Shape:~%")
    (maphash #'(lambda (k v)
		 (format out "    ~a -> ~a~%"
			 (if (symbol-lazyaxis k)
			     (format nil "~a(~a)" k (symbol-lazyaxis k))
			     k)
			 v))
	     cl-waffe2/vm.generic-tensor::*adjustable-shape-table*)
    (format out "LazyAxis Table:~%")
    (maphash #'(lambda (s axis)
		 (format out "    ~a -> ~a~%" s axis))
	     *symbol->lazyaxis*)))

