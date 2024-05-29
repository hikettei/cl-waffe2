(cl:push (cl:pathname "./") ql:*local-project-directories*)
(asdf:defsystem "aten"
  :author "hikettei <ichndm@gmail.com>"
  :licence "MIT"
  :description "Aten Backend for cl-waffe2"
  :serial t
  :depends-on ("flexi-streams" "abstracttensor")
  :components ((:file "package")))
