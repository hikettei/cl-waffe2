
(in-package :cl-waffe2/vm.generic-tensor)

(defun deterministic-p (tensor)
  "Returns t if tensor's node is deterministic
[Any-Previous-Node]
    |
[AnyNode] <- The Given Tensor
    |"
  (declare (type AbstractTensor tensor))
  (= (length (tensor-variables tensor)) 1))

(defun non-deterministic-p (tensor)
  "Returns t if tensor's node is non-deterministic
[Node1] [Node2] ...
    |------|
[AnyNode] <- The Given Tensor
    |"
  (declare (type AbstractTensor tensor))
  (> (length (tensor-variables tensor)) 1))

(deftype node-state-t ()
  "The type node-state-t indicates the keywords that used to express node's transmission state."
  `(member :deterministic :non-deterministic))

(declaim (ftype (function (AbstractTensor) node-state-t) node-state))
(defun node-state (tensor)
  (if (deterministic-p tensor)
      :deterministic
      :non-deterministic))

(defun movetensor-p (node)
  (or (subtypep (class-of node) (find-class 'cl-waffe2/base-impl:MoveTensorNode))
      (subtypep (class-of node) (find-class 'cl-waffe2/base-impl::MoveScalarTensorNode))))

(defmacro ignore-me? (node)
  `(cl-waffe2/base-impl:movetensor-ignore-me ,node))

(defun tensor-attribute (tensor)
  "Return: (member :chain :input)"
  (declare (type AbstractTensor tensor))
  (let ((name (tensor-name tensor)))
    (typecase name
      (string
       (if (eql (tensor-facet tensor) :input)
	   :chain
	   :input)) ;; :chain = auto-generated
      (T :input))))

;; Unused:
(defun trace-and-explore-nodes! (out-tensor)
  "Incf tensor-ref-n
tensor-ref-n indicates that how many times the tensor was used in the node."
  (declare (type AbstractTensor out-tensor)
	   (optimize (speed 3)))
  (mapc
   #'(lambda (tensor)
       (incf (the fixnum (tensor-n-ref tensor)) 1)
       (trace-and-explore-nodes! tensor))
   (tensor-variables out-tensor)))

(defun trace-and-optimize-node! (out-tensor major n-cores)
  "TODO: DOC"
  (declare (type AbstractTensor out-tensor)
	   (type fixnum n-cores)
	   (optimize (speed 3))
	   (type (and keyword (member :speed :memory)) major))

  ;; TODO: (setf lparallel:*kernel* (make-kernel 4))

  ;; MoveTensor(Input/Parameter, ChainTMP) <- COPY it.
  ;; MoveTensor.n_ref = 1, 0 <- DONT COPY IT.
  
  (let* ((current-node   (tensor-backward out-tensor))
 	 (past-variables (tensor-variables out-tensor)))

    (when (and (movetensor-p current-node)
	       ;; [MoveTensor] -> [AnyTensor save-for-backward=t]
	       ;; ↑Ignored.

	       (not (tensor-protect-me (car past-variables)))
	       (not (cl-waffe2/base-impl:movetensor-save-for-backward current-node))
	       (or *no-grad*
		   ;; TODO: Simplify save-for-backward, (compile statically workign kernel?)
		   (ancestor-param-p (car past-variables)))
	       
	       ;; (!copy place past-out) i.e. (!copy Chain Past-Out)

	       ;; The problem is that: it is unknown wheter movetensor returns Viewed Input or not.
	       ;; So Tensors whose place has multi-dimensional offset, is ignored

	       (apply #'order-reductable-p 0 past-variables)
	       
	       (eql (tensor-attribute (car past-variables)) :chain)
	       (let* ((prev-out (second past-variables))
		      (attr     (tensor-attribute prev-out)))
		 (and (<= (the fixnum (tensor-n-ref prev-out)) 1)
		      ;; prev-out is deterministic
		      (not (eql attr :input)))))
      (setf (ignore-me? current-node) t))

    (mapc
     #'(lambda (tensor)
	 (trace-and-optimize-node! tensor major n-cores))
     past-variables)))

(defun optimize-computation-node! (out-tensor major n-cores)
  "The function optimize-computation-node! do these works:

1. Optimize MoveTensorNode
2. Optimize the connection of ChainTMP
3. Scheduling the lparallel depending on their nodes and threads.

Computation Time: O(total_nodes * 2)"
  (declare (type AbstractTensor out-tensor)
	   (type (and keyword (member :speed :memory)) major)
	   (type fixnum n-cores)
	   (optimize (speed 3)))
  
  (when (tensor-traced-p out-tensor)
    (return-from optimize-computation-node!))
  
  (trace-and-explore-nodes! out-tensor)
  (trace-and-optimize-node! out-tensor major n-cores)

  (setf (tensor-traced-p out-tensor) t))

