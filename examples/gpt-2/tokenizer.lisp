
(in-package :gpt-2-example)


;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  Tokenizers
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;; Reference: https://github.com/graykode/gpt-2-Pytorch/blob/master/GPT2/encoder.py
(defparameter *encoder-json* nil)
(defparameter *decoder-json* nil)

(defparameter *pat* (create-scanner "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"))

(defparameter *bpe-merges* nil)

(defun load-bpe-merges (&key (save-path "./examples/gpt-2/assets/models/gpt-2-117M/vocab.bpe"))
  (let* ((bpe (uiop:read-file-string save-path))
	 (bpe (subseq bpe 1 (1- (length bpe))))
	 (out (make-hash-table :test #'equal))
	 (pairs (cdr (loop for mstr in (split "\\n" bpe) collect (split " " mstr)))))
    (loop for p in pairs
	  for i upfrom 0 do
	    (setf (gethash p out) i))
    (setf *bpe-merges* out)
    t))

(defun load-encoder-json (&key (save-path "./examples/gpt-2/assets/models/gpt-2-117M/encoder.json"))
  (format t "[INFO] Loading encoder.json ...~%")
  (let ((encoder-str (time (parse (uiop:read-file-string save-path))))
	(dict (make-hash-table :test #'equal))
	(dec-dict (make-hash-table)))
    (format t "[INFO] Parsing was done... n_vocab=~a~%" (/ (length encoder-str) 2))
    (loop while encoder-str do
      (let ((key (pop encoder-str))
	    (val (pop encoder-str)))
	(setf (gethash val dec-dict) (format nil "~a" key))
	(setf (gethash (format nil "~a" key) dict) val)))
    (setf *decoder-json* dec-dict)
    (setf *encoder-json* dict)))

(defun get-pairs (token)
  (declare (type string token))

  ;; token ... Hi Gthere, ...
  (loop for index fixnum upfrom 0 below (1- (length token))
	collect
	(list (string (aref token index)) (string (aref token (1+ index))))))

(defun countup-nth (word token n)
  (let ((count 0)
	(n (1+ n)))
    (loop for tkn in token
	  for pos upfrom 0
	  if (equal tkn word)
	    do (incf count 1)
	  if (= count n)
	    do (return-from countup-nth pos))))

(defun bpe-split (token)
  (declare (type string token))
  (let ((word (list token))
	(out-of-range (* -1 (+ 1 (length (hash-table-keys *bpe-merges*)))))
	(pairs (get-pairs token)))

    (loop named bpe-iter while t do
      (let* ((smallest (loop for pair in pairs minimize (or (gethash pair *bpe-merges*) out-of-range)))
	     (bigram   (find smallest pairs :test #'eql :key #'(lambda (x) (gethash x *bpe-merges*)))))
	(when (null bigram)
	  (return-from bpe-iter))

	(multiple-value-bind (first second) (apply #'values bigram)
	  (let ((new-word)
		(i 0))
	    (loop named bpe-word-iter while (< i (length word)) do
	      (if (or (null (find first word :test #'equal))
		      (not (< i (count first word :test #'equal))))
		  (progn
		    ;; Break
		    (setq new-word
			  `(,@new-word
			    ,@(subseq word i (length word))))
		    (return-from bpe-word-iter))
		  (let ((j (countup-nth first word i)))
		    (setq new-word
			  `(,@new-word
			    ,@(subseq word i j)))
		    (setq i j)
		    (if (and (equal (nth i word) first)
			     (< i (1- (length word)))
			     (equal (nth (1+ i) word) second))
			(progn
			  (setq new-word
				`(,@new-word
				  ,(concatenate 'string first second)))
			  (incf i 2))
			(progn
			  (setq new-word
				`(,@new-word ,(nth i word)))
			  (incf i 1))))))
	    (setq word new-word)
	    (if (= (length word) 1)
		(return-from bpe-iter)
		(setq pairs (get-pairs word)))))))
    word))


(defun encode-sentence (sentence) ;; (read-line)
  (declare (type string sentence))
  (let ((tokens (all-matches-as-strings *pat* sentence))
	(bpe-tokens))
    (loop for token in tokens do
      (let* ((token (loop for n upfrom 0 below (length token)
			  collect (gethash (char-code (aref token n)) *byte2unicode*)))
	     (token (apply #'concatenate 'string token)))
	(dolist (bpetoken (bpe-split token))
	  (push (+ 0.0 (or (gethash bpetoken *encoder-json*) 0)) bpe-tokens))))
    (let ((tokens (reverse bpe-tokens)))
      (change-facet
       (make-array `(1 ,(length tokens))
		   :element-type 'single-float
		   :initial-contents `(,tokens))
       :direction 'AbstractTensor))))

(defun decode-sentence (list)
  (declare (type list list))
  (let ((text (apply #'concatenate 'string (loop for token in list collect (gethash token *decoder-json*)))))
    (with-output-to-string (out)
      (loop for pos fixnum upfrom 0 below (length text) do
	(let ((code (gethash (char-code (aref text pos)) *byte2unicode*)))
	  (if code
	      (princ code out)
	      (princ " " out)))))))

