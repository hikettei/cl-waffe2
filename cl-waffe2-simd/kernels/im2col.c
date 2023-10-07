
#include "im2col.h"

static void im2col_s_column_major
(
 const float* data_im,
 const uint64_t N,
 const uint64_t C,
 const uint64_t H,
 const uint64_t W,
 const uint64_t H_out,
 const uint64_t W_out,
 const uint64_t K_H,
 const uint64_t K_W,
 const uint64_t pad_h,
 const uint64_t pad_w,
 const uint64_t stride_h,
 const uint64_t stride_w,
 const uint64_t dilation_h,
 const uint64_t dilation_w,
 float* data_col) {
  uint64_t stride_imN = C * H * W;
  uint64_t stride_imC = H * W;
  uint64_t stride_imH = W;
  uint64_t stride_imW = 1;

  // COL1 = (N C K-H K-W H-out W-out)
  uint64_t stride_col1 = C * K_H * K_W * H_out * W_out;
  uint64_t stride_col2 = K_H * K_W * H_out * W_out;
  uint64_t stride_col3 = K_W * H_out * W_out;
  uint64_t stride_col4 = H_out * W_out;
  uint64_t stride_col5 = W_out;
  uint64_t stride_col6 = 1;
  
  for (uint64_t batch_n=0; batch_n < N; batch_n++) {
    for (uint64_t h_start=0; h_start < K_H; h_start++) {
      uint64_t h_end = dilation_h * h_start - pad_h + stride_h * H_out;
      for (uint64_t w_start=0; w_start < K_W; w_start++) {
	uint64_t w_end = dilation_w * w_start - pad_w + stride_w * W_out;
	for (uint64_t H_out_col=0, H_out_im=h_start;
	     H_out_im < h_end;
	     H_out_col++, H_out_im+=stride_h) {
	  for (uint64_t W_out_col=0, W_out_im=w_start;
	       W_out_im < w_end;
	       W_out_col++, W_out_im+=stride_w) {
	    for (uint64_t cth=0; cth < C; cth++) {
	      uint64_t index_im = stride_imN * batch_n + stride_imC * cth + stride_imH * H_out_im + stride_imW * W_out_im;
	      uint64_t index_col = stride_col1 * batch_n + stride_col2 * cth + stride_col3 * h_start + stride_col4 * w_start + stride_col5 * H_out_im + stride_col6 * W_out_im;
	      data_col[index_col] = data_im[index_im];	      
	    }
	  }    
	}
      }
    }
  }  
}
