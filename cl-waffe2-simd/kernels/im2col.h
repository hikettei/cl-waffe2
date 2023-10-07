
#define define_im2col(opname, dtype, define_as, order)			\
  void waffe2_##opname##_##define_as##_##order(const dtype* data_im,	\
					       const uint64_t N,	\
					       const uint64_t C,	\
					       const uint64_t H,	\
					       const uint64_t W,	\
					       const uint64_t H_out,	\
					       const uint64_t W_out,	\
					       const uint64_t K_H,	\
					       const uint64_t K_W,	\
					       const uint64_t pad_h,	\
					       const uint64_t pad_w,	\
					       const uint64_t stride_h,	\
					       const uint64_t stride_w,	\
					       const uint64_t dilation_h, \
					       const uint64_t dilation_w, \
					       dtype* data_col);	\

define_im2col(im2col, float, s, column);
define_im2col(im2col, double, d, column);

