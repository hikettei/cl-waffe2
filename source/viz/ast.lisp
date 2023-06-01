
(in-package :cl-waffe2/viz)

(defstruct (AstNode
	    (:constructor
		make-astnode (tensor)))
  (id (gensym "NODE"))
  (n-ref (cl-waffe2/vm.generic-tensor::tensor-n-ref tensor))
  (tensor tensor)
  (node (tensor-backward tensor))
  (shape (shape tensor) :type list))

(defmethod print-object ((object AstNode) stream)
  (format stream "<Node: ~a ~a>"
	  (or (astnode-node object)
	      (if (tensor-name (astnode-tensor object))
		  (format nil "{InputTensor ~a}" (tensor-name (astnode-tensor object)))
		  "{:Parameter}"))
	  (astnode-shape object)))

(defun node-attribute (node)
  (declare (type astnode node))
  (or (and (astnode-node node) :node)
      (let ((name (tensor-name (astnode-tensor node))))
	(typecase name
	  (string :chain)
	  (keyword :input)
	  (T :parameter)))))

(defun movetensor-p (node)
  (subtypep (class-of node)
	    'cl-waffe2/base-impl:MoveTensorNode))

(defparameter *all-of-nodes* nil)

(defun trace-nodes (tensor)
  (declare (type AbstractTensor)
	   (optimize (speed 3)))
  (cons (let ((res (make-astnode tensor)))
	  (push res *all-of-nodes*)
	  res)
	(map 'list #'trace-nodes (tensor-variables tensor))))


(defun viz-computation-node (out-tensor
			     stream
			     &key
			       (format :dot))
  "TODO: DOCSTRING"
  (declare (type AbstractTensor out-tensor)
	   (type (and keyword (member :dot :print)) format))
  (case format
    (:dot
     (output-to-dot out-tensor stream))
    (:print
     (output-to-stream out-tensor stream))))


(defun make-ast-dot (tensor)
  (let* ((*all-of-nodes*)
	 (ast (trace-nodes tensor)))
    (values ast (copy-list *all-of-nodes*))))

(defun node-dot-id (node)
  (let ((type (node-attribute node)))
    (case type
      (:node (astnode-id node))
      (t     (tensor-name (astnode-tensor node))))))

(defun node-color (node)
  (let ((type (node-attribute node)))
    (if (movetensor-p (astnode-node node))
	"gray"
	(case type
	  (:node "#e6e6fa")
	  (:chain "#f0e68c")
	  (:input "#f0f8ff")
	  (:parameter "#ff6347")))))

(defun node-address-id (node)
  #+sbcl(format nil "~%~x" (sb-kernel:get-lisp-obj-address node))
  #-sbcl "")

(defun node-dot-name (node)
  (let ((type (node-attribute node)))
    (case type
      (:node (let* ((name (symbol-name (class-name (class-of (astnode-node node)))))
		    (pos (or (position #\- name :test #'equal) (length name))))
	       (if (equal "MOVETENSORNODE" (subseq name 0 pos))
		   (format nil "~a"
			   (if (movetensor-ignore-me (astnode-node node))
			       "(Deleted)"
			       "Move"))
		   (format nil "~a [~a]" (subseq name 0 pos) (astnode-n-ref node)))))
      (t
       (let ((tensor (astnode-tensor node)))
	 (format nil
		 "~a~%~a"
		 (tensor-name tensor)
		 (shape tensor)))))))

(defun determine-style (x y)
  (let ((x-type (node-attribute x))
	(y-type (node-attribute y)))

    (cond
      ((or (find x-type `(:chain :input :parameter))
	   (find y-type `(:chain :input :parameter)))
       "[style=\"dashed\"]")
      ((and (not (equal "Move" (node-dot-name x)))
	    (not (equal "Move" (node-dot-name y))))
       "[penwidth=\"2\"]")
      (T "[weight=10]"))))

;; dot -Tpng ./out.dot > ./out.png
(defun output-to-dot (out-tensor filepath)
  (multiple-value-bind (ast nodes) (make-ast-dot out-tensor)
    (with-open-file (stream
		     filepath
		     :direction :output
		     :if-exists :supersede
		     :if-does-not-exist :create)
      (format stream "digraph computation_node {~%  node[shape=\"box\" style=\"filled\" color=\"black\" penwidth=\"2\"];~%")
      
      (dolist (n nodes)
	(format stream "  ~a [label = \"~a\" fillcolor=\"~a\" style=\"filled, solid\"];~%"
		(node-dot-id n)
		(node-dot-name n)
		(node-color n)))

      (labels ((explore (toplevel)
		 (let* ((top  (car toplevel))
			(rest (cdr toplevel)))
		   (dolist (r rest)
		     (format stream "  ~a -> ~a~a;~%"
			     (node-dot-id (car r))
			     (node-dot-id top)
			     (determine-style (car r) top))
		     (explore r)))))
	(explore ast))
      (format stream "}~%"))))


;; Tests (TODO: Delete)
