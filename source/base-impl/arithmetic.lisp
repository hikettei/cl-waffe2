
(in-package :cl-waffe2/base-impl)

(defnode (AddNode (myself)
	  :where `([~] [~] -> [~])
	  :slots nil
	  :documentation "Computes x + y element-wise."))

(defnode (SubNode (myself)
	  :where `([~] [~] -> [~])
	  :slots nil
	  :documentation "Computes x -  y element-wise."))

(defnode (MulNode (myself)
	  :where `([~] [~] -> [~])
	  :slots nil
	  :documentation "Computes x * y element-wise."))

(defnode (DivNode (myself)
	  :where `([~] [~] -> [~])
	  :slots nil
	  :documentation "Computes x / y element-wise."))
