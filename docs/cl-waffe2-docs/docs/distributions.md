
 # Distributions
@begin(section)
@title(Samples matrices from distribution)

In order to create new matrices from distribution, cl-waffe2 provides a package, @c(cl-waffe2/distributions).
@begin(section)
@title(Common Format to the APIs)

All sampling functions are defined in the following format:


via @c(define-tensor-initializer) macro.
@c((function-name shape [Optional Arguments] &rest args &keys &allow-other-keys))
@c(args) is a arguments which passed to make-tensor, accordingly, both of these functions are valid for example.
@begin(enum)
@item((normal `(10 10) 0.0 1.0 :dtype :double))
@item((normal `(10 10) 0.0 1.0 :requires-grad t))
@end(enum)
@end(section)
@end(section)
@begin(section)
@title(define-tensor-initializer)

@cl:with-package[name="cl-waffe2/distributions"](
@cl:doc(macro define-tensor-initializer)
)

@end(section)
@begin(section)
@title(ax+b)

@cl:with-package[name="cl-waffe2/distributions"](
@cl:doc(function ax+b)
)

@end(section)
@begin(section)
@title(beta)

@cl:with-package[name="cl-waffe2/distributions"](
@cl:doc(function beta)
)

@end(section)
@begin(section)
@title(bernoulli)

@cl:with-package[name="cl-waffe2/distributions"](
@cl:doc(function bernoulli)
)

@end(section)
@begin(section)
@title(chisquare)

@cl:with-package[name="cl-waffe2/distributions"](
@cl:doc(function chisquare)
)

@end(section)
@begin(section)
@title(expotential)

@cl:with-package[name="cl-waffe2/distributions"](
@cl:doc(function expotential)
)

@end(section)
@begin(section)
@title(gamma)

@cl:with-package[name="cl-waffe2/distributions"](
@cl:doc(function gamma)
)

@end(section)
@begin(section)
@title(normal)

@cl:with-package[name="cl-waffe2/distributions"](
@cl:doc(function normal)
)

@end(section)
@begin(section)
@title(uniform-random)

@cl:with-package[name="cl-waffe2/distributions"](
@cl:doc(function uniform-random)
)

@end(section)
@begin(section)
@title(randn)

@cl:with-package[name="cl-waffe2/distributions"](
@cl:doc(function randn)
)

@end(section)
@end(section)