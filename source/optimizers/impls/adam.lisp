
(in-package :cl-waffe2/optimizers)


(defoptimizer (Adam (self param
			  &key
			  (lr 1e-3)
			  (eps 1e-7)
			  (beta1 0.9)
			  (beta2 0.999))
		    :documentation "
### Inputs

Implements Adam algorithm.

See the [original paper](https://arxiv.org/abs/1412.6980) for detailed algorithms.
"
		    :slots ((lr :initarg :lr :reader lr-of :type single-float)
			    (eps :initarg :eps :reader eps-of :type single-float)
			    (beta1 :initarg :beta1 :reader beta1-of :type single-float)
			    (beta2 :initarg :beta2 :reader beta2-of :type single-float)
			    (N :initform 0 :type fixnum :accessor adam-n)
			    (M :accessor adam-m)
			    (V :accessor adam-V)))
    

    (setf (adam-m self)
	  (make-tensor (shape param)
		       :dtype (dtype param)
		       :order (order param)))

    (setf (adam-v self)
  	  (make-tensor (shape param)
		       :dtype (dtype param)
		       :order (order param))))

(defmethod cl-waffe2/vm.nodes:on-print-object ((opt Adam) stream)
  (format stream "lr=~a eps=~a beta1=~a beta2=~a N=~a"
	  (lr-of opt)
	  (eps-of opt)
	  (beta1-of opt)
	  (beta2-of opt)
	  (adam-n opt)))

(defmodel (Adam-Step-M (self)
	         ;;  M    Grad       beta1
	   :where (M[~] X.grad[~] decay-rate[scal] -> M[~] where scal = 1)
	   :on-call-> ((self m x-grad decay-rate)
		       (declare (ignore self))
		       (A+=B m (!mul (!sub x-grad m)
				     (!sub 1.0 decay-rate))))))

(defmodel (Adam-Step-V (self)
	   ;;      V      Grad      beta2
	   :where (V[~] X.grad[~] decay-rate[scal] -> V[~] where scal = 1)
	   :on-call-> ((self v x-grad decay-rate)
		       (declare (ignore self))
		       (A+=B v (!mul (!sub 1.0 decay-rate)
				     (!sub (!mul x-grad x-grad) v))))))

(defmodel (Adam-Step-Param (self)
	   :where (M[~] V[~] Param[~] Lr-t[scal] Eps[scal] -> Param[~] where scal = 1)
	   :on-call-> ((self m v param lr-t eps)
		       (declare (ignore self))
		       (A-=B param
			     (!div (!mul m lr-t)
				   (!add eps (!sqrt v)))))))

(defmodel-as (Adam-Step-M)
  :asif :function
  :named apply-adam-step-m)

(defmodel-as (Adam-Step-V)
  :asif :function
  :named apply-adam-step-v)

(defmodel-as (adam-Step-Param)
  :asif :function
  :named apply-adam-step-param)

(defun adam-step-lr (adam)
  (declare (type Adam adam)
	   (optimize (speed 3)))
  (incf (the fixnum (adam-n adam)) 1)
  (let ((n  (adam-n adam))
	(lr (lr-of adam))
	(b1 (beta1-of adam))
	(b2 (beta2-of adam)))
    (declare (type single-float lr b1 b2)
	     (type fixnum n))

    (* lr
       (/ (sqrt (the (single-float 0e0) (- 1.0 (expt b2 n))))
	  (- 1.0 (expt b1 n))))))

(defmethod step-optimize ((optimizer Adam))
  (let* ((lr-t (adam-step-lr optimizer))
	 (param (read-parameter optimizer))
	 (grad  (grad param)))
    (with-no-grad
      (apply-adam-step-m
       (adam-m optimizer)
       grad ;; the gradient involved in in-place op
       (make-tensor (beta1-of optimizer)))
      
      (apply-adam-step-v
       (adam-v optimizer)
       grad ;; never destruct the gradients
       (make-tensor (beta2-of optimizer)))

      (apply-adam-step-param
       (adam-m optimizer)
       (adam-v optimizer)
       param
       (make-tensor lr-t) ;; Runtime creation of ScalarTensor ... takes a little overhead (approx: 0.00013 sec * N_parameters)
       (make-tensor (eps-of optimizer))))))

