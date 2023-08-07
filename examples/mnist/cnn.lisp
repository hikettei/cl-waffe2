
(in-package :mnist-sample)

(defsequence CNN ()
	     (Conv2D 3 16  `(3 3))
	     (asnode #'!relu)     
	     (MaxPool2D    `(2 2))
	     (Conv2D 16 32 `(5 5))
	     (asnode #'!relu)
	     (MaxPool2D `(2 2))
	     (asnode #'lazy-print)
	     (asnode #'!reshape t (* 32 5 5)) 
	     (LinearLayer (* 32 5 5) 10))

