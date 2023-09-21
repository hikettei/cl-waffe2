
(in-package :gpt-2-example)

;; Utils

(defun load-npy (path &rest args)
  ;; npz -> AbstractTensor
  (format t "[INFO] load-npy attempts to load ~a...~%" (apply #'format nil path args))
  (parameter (change-facet (numpy-file-format:load-array (apply #'format nil path args)) :direction 'AbstractTensor)))

(defun read-symbol (a) (cl-waffe2/vm.generic-tensor::read-symbol a))

(defun incf-tensor-ptr (tensor tensor-ptr &key (offset 0))
  (cl-waffe2/backends.cpu::incf-tensor-ptr tensor tensor-ptr :offset offset))

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  [AbstractNode] GPT2PositionalEmbedding
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(defun expand-embedding-form (ctx wte wpe wte-ptr wpe-ptr ctx-out-ptr ctx-out ctx-view ctx-out-view)
  ;; Returns a S-expression later compiled
  ;; CTX = [~ sentence-length], Sparse Matrix
  ;; WTE = [Vocab-Size Embedding-Size]
  ;; WPE = [N-CTX Embedding-Size]
  (declare (type AbstractTensor ctx wte wpe ctx-out))

  (let ((embedding-size (second (shape wte)))
	(iternum (if (numberp (second (shape ctx)))
		     (second (shape ctx))
		     `(read-symbol ',(second (shape ctx))))))
    (with-gensyms (position-n vocab-index wte-position wpe-position)
      `(progn
	 (cl-waffe2/backends.cpu::waffe2-smul-scal
	  ,embedding-size
	  (incf-tensor-ptr ,ctx-out ,ctx-out-ptr :offset ,(offset-of ctx-out-view 0))
	  1
	  0.0)
	 (loop for ,position-n fixnum upfrom 0 below ,iternum do
	   (let* ((,vocab-index (aref (tensor-vec ,ctx) (+ ,position-n ,(offset-of ctx-view 0))))
		  (,wte-position (* (round (the single-float ,vocab-index)) ,embedding-size))
		  (,wpe-position (* ,position-n ,embedding-size)))
	     ;; ctx-out <- add(WTE[Word_Index, :], WPE[Position, :])

	     ;; [TODO] Fuse these steps to get more speed:
	     ;; Using SIMD Extention to add two vectors
	     ;; sadd: Y += X

	     ;; Y *= 0
	     ;; Y += WTE
	     ;; Y += WPE
	     (cl-waffe2/backends.cpu::waffe2-sadd
	      ,embedding-size	      
	      (incf-tensor-ptr ,ctx-out ,ctx-out-ptr :offset (+ ,(offset-of ctx-out-view 0)
								(* ,position-n ,embedding-size))) ;; CTX-OUT[:, pos, embedding-size]
	      1
	      (incf-tensor-ptr ,wte ,wte-ptr :offset ,wte-position)
	      1)

	     (cl-waffe2/backends.cpu::waffe2-sadd
	      ,embedding-size
	      (incf-tensor-ptr ,ctx-out ,ctx-out-ptr :offset (+ ,(offset-of ctx-out-view 0)
								(* ,position-n ,embedding-size))) ;; CTX-OUT[:, pos, embedding-size]
	      1
	      (incf-tensor-ptr ,wpe ,wpe-ptr :offset ,wpe-position)
	      1)))))))

;; Implementing Embedding
;; [TODO] Backward, Include this to :cl-waffe2/nn package
(defnode (GPT2PositionalEmbedding (self vocab-size n-ctx embedding-size)
	  :documentation "add(WTE[CTX], WPE[CTX]) -> CTX"
	  :where (ctx[N sentence-length] wte[vocab-size embedding-size] wpe[n-ctx embedding-size] ctx-out[N sentence-length embedding-size]
			->
			ctx-out[N sentence-length embedding-size])))

(define-impl (GPT2PositionalEmbedding :device LispTensor :cache-when-compiled nil)
	     :forward ((self ctx wte wpe ctx-out)

		       ;; This is intended because I want to explict the reason of error
		       (when (cl-waffe2/backends.cpu::simd-extension-p)
			 (error "GPT2 Example seems working without SIMD-Extension. Because GPT2PositionalEmbedding depends on foreign simd library, SIMD-Extension must be loaded in advance.

You can simply run:
    $ make build_simd_extension

In your terminal, and cl-waffe2 will load it."))

		       (assert
			(and (eql (order ctx) :column)
			     (eql (order wpe) :column)
			     (eql (order wte) :column))
			nil
			"GPT2PositionalEmbedding: Orders must be :column (C Order), not a :row (Fortran Order)")

		       (with-gensyms (wte-ptr wpe-ptr ctx-out-ptr)
			 `(locally (declare (optimize (speed 1)))
			    (cl-waffe2/backends.cpu::with-tensor-ptrs ((,wpe-ptr ,wpe)
								       (,wte-ptr ,wte)
								       (,ctx-out-ptr ,ctx-out))
			      (,@(call-with-view
				  #'(lambda (ctx-view ctx-out-view)
				      (expand-embedding-form ctx wte wpe wte-ptr wpe-ptr ctx-out-ptr ctx-out ctx-view ctx-out-view))
				  (list ctx ctx-out)
				  :at-least-dim 1
				  :force-order t
				  :lparallel nil
				  :fuse nil)
			       ,ctx-out))))))

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  Defines Activations
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

;; [TODO] Move this into :cl-waffe2/nn package
;; eps=1.0e-5, [FixME] rounding error of mean
(defun LayerNorm-Revisit (x g b &key (eps 1.0e-5))
  ;; X ... [N, Sentene_length, Embedding_DIM]
  ;; g/b... [Embedding_DIM]

  ;; (!sub x u) != 0.0 ...
  (let* ((u (!mean x :axis -1 :keepdims t))  ;; μ=mean(x, axis=-1)
	 (s (!mean (!expt (!sub x u) 2) :axis -1 :keepdims t))
	 (x (!div (!sub x u)
		  (!sqrt (!add (->contiguous s) eps)))))
    (!add
     (!mul
      x
      (%transform g[i] -> g[~ i]))
     (%transform b[i] -> b[~ i]))))

(defun !gelu-lisptanh (x)
  (!* 0.5 x
      (!+ 1
	  (let ((o (!* (coerce (sqrt (/ 2.0 pi)) (dtype->lisp-type (dtype x)))
		       (!+ x
			   (!* 0.044715 (!expt x 3))))))
	    (with-devices (LispTensor)
	      (!tanh o))))))

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  Defines more utils
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(defun !affine (x weight bias)
  (!add (!matmul (!t x) (!flexible weight))
	(%transform bias[i] -> bias[~ i])))

(defmacro asSetq ((&rest out-binds) func &rest inputs &aux (out (gensym)) (x (gensym)))
  "out-binds = (asnode lambda x: (funcall func x inputs...))"
  `(asnode
    #'(lambda (,x)
	(let ((,out (multiple-value-list (funcall ,func ,x ,@inputs))))
	  ,@(loop for out in out-binds
		  for nth upfrom 0
		  if out
		    collect `(setq ,out (nth ,nth ,out)))
	  (apply #'values ,out)))))

;; not working
(defun !cat (a b
	     &key (dim 0)
	     &aux (dim (if (>= dim 0)
			   dim
			   (+ dim (dims a)))))
  (let ((a-views (loop for nth upfrom 0
		       for a   in (shape a)
		       if (= nth dim)
			 collect `(0, a)
		       else
			 collect t))
	(b-views (loop for nth upfrom 0
		       for a   in (shape a)
		       for b   in (shape b)
		       if (= nth dim)
			 collect `(,a ,(+ a b))
		       else
			 collect t))		       
	(out (make-input (loop for nth upfrom 0
			       for a in (shape a)
			       for b in (shape b)
			       if (= nth dim)
				 collect (+ a b)
			       else
				 collect (progn
					   (assert (= a b) nil "!cat: Assertion Failed")
					   a))
			 nil
			 :dtype (dtype a))))
    (call-> out
	    (apply #'asnode #'!view a-views)
	    (asnode #'!move a)
	    (asnode #'!view)
	    (apply #'asnode #'!view b-views)
	    (asnode #'!move b)
	    (asnode #'!view t))))


;; Known issue:
;; SLEEF stanh overflows under |x| > 10000.0 range...
;; GeLU but tanh is safe

;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
;;  MHA
;; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(defun split-heads (x qkv &aux (stride (/ (car (last (shape x))) 3)))
  (declare (type (member :Q :K :V) qkv))
  ;; X ... [~ Sentence_length 2304] -> Q[~ Sentece-Length 786] K V...
  (let* ((x (!view x t t (case qkv
			   (:Q `(0 ,stride))
			   (:K `(,stride ,(* 2 stride)))
			   (:V `(,(* 2 stride) ,(* 3 stride))))))
	 (new-x-shape `(,@(butlast (shape x)) ,(read-config :n-head) ,(/ (car (last (shape x))) (read-config :n-head))))
	 (out          (apply #'!reshape x new-x-shape)))
    (if (eql qkv :K)
	(!permute out (torch-order 0 2 3 1)) ;; (batch head head-features N)
	(!permute out (torch-order 0 2 1 3))))) ;; (batch head-f N head)

(defun merge-heads (x)
  (let* ((x (!permute x (torch-order 0 2 1 3)))
	 (x-shape `(,@(butlast (shape x) 2) ,(apply #'* (last (shape x) 2)))))
    (apply #'!reshape x x-shape)))

(defun SelfAttention (x past) ;; X ...[N 2304]
  (let* ((K (split-heads x :K))
	 (V (split-heads x :V))
	 (Q (split-heads x :Q)))

    (when past
      (let ((pkey (car past))
	    (pval (second past)))
	(setq K (!cat pkey (!t K) :dim -1))
	(setq V (!cat pval V :dim -2))))
    
    (let ((w (!matmul Q K)))
      (values
       (call-> w
	       (asnode #'!div (make-tensor (car (last (shape w)))))
	       (asnode #'!softmax :avoid-overflow nil)
	       (asnode #'!matmul v)
	       (asnode #'merge-heads))
       (list K V)))))

;; ~~~~~~~~~~~~~~~~~~~~~~~
;; Constant Table for BPE
;; ~~~~~~~~~~~~~~~~~~~~~~~

(defun list->hash-table (list)
  (let ((out (make-hash-table)))
    (loop for kv-pair in list do
      (setf (gethash (car kv-pair) out) (second kv-pair)))
    out))

(defun hash-table-revkv (hash-table)
  (let ((out (make-hash-table :test #'equal)))
    (loop for key in (hash-table-keys hash-table) do
      (setf (gethash (gethash key hash-table) out) key))
    out))

(defparameter *byte2unicode*
  (list->hash-table
  `((33 "!")
    (34 "\"")
    (35 "#")
    (36 "$")
    (37 "%")
    (38 "&")
    (39 "'")
    (40 "(")
    (41 ")")
    (42 "*")
    (43 "+")
    (44 ",")
    (45 "-")
    (46 ".")
    (47 "/")
    (48 "0")
    (49 "1")
    (50 "2")
    (51 "3")
    (52 "4")
    (53 "5")
    (54 "6")
    (55 "7")
    (56 "8")
    (57 "9")
    (58 ":")
    (59 ";")
    (60 "<")
    (61 "=")
    (62 ">")
    (63 "?")
    (64 "@")
    (65 "A")
    (66 "B")
    (67 "C")
    (68 "D")
    (69 "E")
    (70 "F")
    (71 "G")
    (72 "H")
    (73 "I")
    (74 "J")
    (75 "K")
    (76 "L")
    (77 "M")
    (78 "N")
    (79 "O")
    (80 "P")
    (81 "Q")
    (82 "R")
    (83 "S")
    (84 "T")
    (85 "U")
    (86 "V")
    (87 "W")
    (88 "X")
    (89 "Y")
    (90 "Z")
    (91 "[")
    (92 "\\")
    (93 "]")
    (94 "^")
    (95 "_")
    (96 "`")
    (97 "a")
    (98 "b")
    (99 "c")
    (100 "d")
    (101 "e")
    (102 "f")
    (103 "g")
    (104 "h")
    (105 "i")
    (106 "j")
    (107 "k")
    (108 "l")
    (109 "m")
    (110 "n")
    (111 "o")
    (112 "p")
    (113 "q")
    (114 "r")
    (115 "s")
    (116 "t")
    (117 "u")
    (118 "v")
    (119 "w")
    (120 "x")
    (121 "y")
    (122 "z")
    (123 "{")
    (124 "|")
    (125 "}")
    (126 "~")
    (161 "¡")
    (162 "¢")
    (163 "£")
    (164 "¤")
    (165 "¥")
    (166 "¦")
    (167 "§")
    (168 "¨")
    (169 "©")
    (170 "ª")
    (171 "«")
    (172 "¬")
    (174 "®")
    (175 "¯")
    (176 "°")
    (177 "±")
    (178 "²")
    (179 "³")
    (180 "´")
    (181 "µ")
    (182 "¶")
    (183 "·")
    (184 "¸")
    (185 "¹")
    (186 "º")
    (187 "»")
    (188 "¼")
    (189 "½")
    (190 "¾")
    (191 "¿")
    (192 "À")
    (193 "Á")
    (194 "Â")
    (195 "Ã")
    (196 "Ä")
    (197 "Å")
    (198 "Æ")
    (199 "Ç")
    (200 "È")
    (201 "É")
    (202 "Ê")
    (203 "Ë")
    (204 "Ì")
    (205 "Í")
    (206 "Î")
    (207 "Ï")
    (208 "Ð")
    (209 "Ñ")
    (210 "Ò")
    (211 "Ó")
    (212 "Ô")
    (213 "Õ")
    (214 "Ö")
    (215 "×")
    (216 "Ø")
    (217 "Ù")
    (218 "Ú")
    (219 "Û")
    (220 "Ü")
    (221 "Ý")
    (222 "Þ")
    (223 "ß")
    (224 "à")
    (225 "á")
    (226 "â")
    (227 "ã")
    (228 "ä")
    (229 "å")
    (230 "æ")
    (231 "ç")
    (232 "è")
    (233 "é")
    (234 "ê")
    (235 "ë")
    (236 "ì")
    (237 "í")
    (238 "î")
    (239 "ï")
    (240 "ð")
    (241 "ñ")
    (242 "ò")
    (243 "ó")
    (244 "ô")
    (245 "õ")
    (246 "ö")
    (247 "÷")
    (248 "ø")
    (249 "ù")
    (250 "ú")
    (251 "û")
    (252 "ü")
    (253 "ý")
    (254 "þ")
    (255 "ÿ")
    (0 "Ā")
    (1 "ā")
    (2 "Ă")
    (3 "ă")
    (4 "Ą")
    (5 "ą")
    (6 "Ć")
    (7 "ć")
    (8 "Ĉ")
    (9 "ĉ")
    (10 "Ċ")
    (11 "ċ")
    (12 "Č")
    (13 "č")
    (14 "Ď")
    (15 "ď")
    (16 "Đ")
    (17 "đ")
    (18 "Ē")
    (19 "ē")
    (20 "Ĕ")
    (21 "ĕ")
    (22 "Ė")
    (23 "ė")
    (24 "Ę")
    (25 "ę")
    (26 "Ě")
    (27 "ě")
    (28 "Ĝ")
    (29 "ĝ")
    (30 "Ğ")
    (31 "ğ")
    (32 "Ġ")
    (127 "ġ")
    (128 "Ģ")
    (129 "ģ")
    (130 "Ĥ")
    (131 "ĥ")
    (132 "Ħ")
    (133 "ħ")
    (134 "Ĩ")
    (135 "ĩ")
    (136 "Ī")
    (137 "ī")
    (138 "Ĭ")
    (139 "ĭ")
    (140 "Į")
    (141 "į")
    (142 "İ")
    (143 "ı")
    (144 "Ĳ")
    (145 "ĳ")
    (146 "Ĵ")
    (147 "ĵ")
    (148 "Ķ")
    (149 "ķ")
    (150 "ĸ")
    (151 "Ĺ")
    (152 "ĺ")
    (153 "Ļ")
    (154 "ļ")
    (155 "Ľ")
    (156 "ľ")
    (157 "Ŀ")
    (158 "ŀ")
    (159 "Ł")
    (160 "ł")
    (173 "Ń"))))

(defparameter *unicode2byte* (hash-table-revkv *byte2unicode*))

