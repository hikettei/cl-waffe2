
#include <stdint.h>

// [fixme] C++ <-> Common Lisp interop is boring 4 me...
// i'm reluctant to manually expanding templates but sadly there's no other option.
// for me it was makeshift solution :(

void waffe2_im2col_d_column
(
 double* data_col,
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
 const double* data_im) {
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
	      uint64_t index_col = stride_col1 * batch_n + stride_col2 * cth + stride_col3 * h_start + stride_col4 * w_start + stride_col5 * H_out_col + stride_col6 * W_out_col;
	      uint64_t index_im = stride_imN * batch_n + stride_imC * cth + stride_imH * H_out_im + stride_imW * W_out_im;	      
	      data_col[index_col] = data_im[index_im];	      
	    }
	  }    
	}
      }
    }
  }  
}


void waffe2_im2col_s_column
(
 float* data_col,
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
 const float* data_im) {
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
	      uint64_t index_col = stride_col1 * batch_n + stride_col2 * cth + stride_col3 * h_start + stride_col4 * w_start + stride_col5 * H_out_col + stride_col6 * W_out_col;
	      uint64_t index_im = stride_imN * batch_n + stride_imC * cth + stride_imH * H_out_im + stride_imW * W_out_im;	      
	      data_col[index_col] = data_im[index_im];	      
	    }
	  }    
	}
      }
    }
  }  
}

void waffe2_im2col_i_column
(
 int* data_col,
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
 const int* data_im) {
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
	      uint64_t index_col = stride_col1 * batch_n + stride_col2 * cth + stride_col3 * h_start + stride_col4 * w_start + stride_col5 * H_out_col + stride_col6 * W_out_col;
	      uint64_t index_im = stride_imN * batch_n + stride_imC * cth + stride_imH * H_out_im + stride_imW * W_out_im;	      
	      data_col[index_col] = data_im[index_im];	      
	    }
	  }    
	}
      }
    }
  }  
}

// Row-Major
void waffe2_im2col_d_row
(
 double* data_col,
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
 const double* data_im) {
  uint64_t stride_imW = C * H * W;
  uint64_t stride_imH = H * W;
  uint64_t stride_imC = W;
  uint64_t stride_imN = 1;

  // COL1 = (N C K-H K-W H-out W-out)
  uint64_t stride_col6 = C * K_H * K_W * H_out * W_out;
  uint64_t stride_col5 = K_H * K_W * H_out * W_out;
  uint64_t stride_col4 = K_W * H_out * W_out;
  uint64_t stride_col3 = H_out * W_out;
  uint64_t stride_col2 = W_out;
  uint64_t stride_col1 = 1;
  
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
	      uint64_t index_col = stride_col1 * batch_n + stride_col2 * cth + stride_col3 * h_start + stride_col4 * w_start + stride_col5 * H_out_col + stride_col6 * W_out_col;
	      uint64_t index_im = stride_imN * batch_n + stride_imC * cth + stride_imH * H_out_im + stride_imW * W_out_im;	      
	      data_col[index_col] = data_im[index_im];	      
	    }
	  }    
	}
      }
    }
  }  
}


void waffe2_im2col_s_row
(
 float* data_col,
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
 const float* data_im) {
  uint64_t stride_imW = C * H * W;
  uint64_t stride_imH = H * W;
  uint64_t stride_imC = W;
  uint64_t stride_imN = 1;

  // COL1 = (N C K-H K-W H-out W-out)
  uint64_t stride_col6 = C * K_H * K_W * H_out * W_out;
  uint64_t stride_col5 = K_H * K_W * H_out * W_out;
  uint64_t stride_col4 = K_W * H_out * W_out;
  uint64_t stride_col3 = H_out * W_out;
  uint64_t stride_col2 = W_out;
  uint64_t stride_col1 = 1;
  
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
	      uint64_t index_col = stride_col1 * batch_n + stride_col2 * cth + stride_col3 * h_start + stride_col4 * w_start + stride_col5 * H_out_col + stride_col6 * W_out_col;
	      uint64_t index_im = stride_imN * batch_n + stride_imC * cth + stride_imH * H_out_im + stride_imW * W_out_im;	      
	      data_col[index_col] = data_im[index_im];	      
	    }
	  }    
	}
      }
    }
  }  
}

void waffe2_im2col_i_row
(
 int* data_col,
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
 const int* data_im) {
  uint64_t stride_imW = C * H * W;
  uint64_t stride_imH = H * W;
  uint64_t stride_imC = W;
  uint64_t stride_imN = 1;

  // COL1 = (N C K-H K-W H-out W-out)
  uint64_t stride_col6 = C * K_H * K_W * H_out * W_out;
  uint64_t stride_col5 = K_H * K_W * H_out * W_out;
  uint64_t stride_col4 = K_W * H_out * W_out;
  uint64_t stride_col3 = H_out * W_out;
  uint64_t stride_col2 = W_out;
  uint64_t stride_col1 = 1;
  
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
	      uint64_t index_col = stride_col1 * batch_n + stride_col2 * cth + stride_col3 * h_start + stride_col4 * w_start + stride_col5 * H_out_col + stride_col6 * W_out_col;
	      uint64_t index_im = stride_imN * batch_n + stride_imC * cth + stride_imH * H_out_im + stride_imW * W_out_im;	      
	      data_col[index_col] = data_im[index_im];	      
	    }
	  }    
	}
      }
    }
  }  
}


// Col2im

void waffe2_col2im_d_column
(
 const double* data_col,
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
 double* data_im) {

  for (uint64_t i=0;i<N * C * H * W;i++) { data_im[i] = 0.0; } 
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
	      uint64_t index_col = stride_col1 * batch_n + stride_col2 * cth + stride_col3 * h_start + stride_col4 * w_start + stride_col5 * H_out_col + stride_col6 * W_out_col;
	      uint64_t index_im = stride_imN * batch_n + stride_imC * cth + stride_imH * H_out_im + stride_imW * W_out_im;	      
	      data_im[index_im] += data_col[index_col];	      
	    }
	  }    
	}
      }
    }
  }  
}


void waffe2_col2im_s_column
(
 const float* data_col,
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
 float* data_im) {
  for (uint64_t i=0;i<N * C * H * W;i++) { data_im[i] = 0.0; } 
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
	      uint64_t index_col = stride_col1 * batch_n + stride_col2 * cth + stride_col3 * h_start + stride_col4 * w_start + stride_col5 * H_out_col + stride_col6 * W_out_col;
	      uint64_t index_im = stride_imN * batch_n + stride_imC * cth + stride_imH * H_out_im + stride_imW * W_out_im;	      
	      data_im[index_im] += data_col[index_col];
	    }
	  }    
	}
      }
    }
  }  
}

void waffe2_col2im_i_column
(
 const int* data_col,
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
 int* data_im) {
  for (uint64_t i=0;i<N * C * H * W;i++) { data_im[i] = 0; } 
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
	      uint64_t index_col = stride_col1 * batch_n + stride_col2 * cth + stride_col3 * h_start + stride_col4 * w_start + stride_col5 * H_out_col + stride_col6 * W_out_col;
	      uint64_t index_im = stride_imN * batch_n + stride_imC * cth + stride_imH * H_out_im + stride_imW * W_out_im;	      
	      data_im[index_im] += data_col[index_col];   
	    }
	  }    
	}
      }
    }
  }  
}

// Row-Major
void waffe2_col2im_d_row
(
 const double* data_col,
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
 double* data_im) {
  for (uint64_t i=0;i<N * C * H * W;i++) { data_im[i] = 0.0; } 
  uint64_t stride_imW = C * H * W;
  uint64_t stride_imH = H * W;
  uint64_t stride_imC = W;
  uint64_t stride_imN = 1;

  // COL1 = (N C K-H K-W H-out W-out)
  uint64_t stride_col6 = C * K_H * K_W * H_out * W_out;
  uint64_t stride_col5 = K_H * K_W * H_out * W_out;
  uint64_t stride_col4 = K_W * H_out * W_out;
  uint64_t stride_col3 = H_out * W_out;
  uint64_t stride_col2 = W_out;
  uint64_t stride_col1 = 1;
  
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
	      uint64_t index_col = stride_col1 * batch_n + stride_col2 * cth + stride_col3 * h_start + stride_col4 * w_start + stride_col5 * H_out_col + stride_col6 * W_out_col;
	      uint64_t index_im = stride_imN * batch_n + stride_imC * cth + stride_imH * H_out_im + stride_imW * W_out_im;	      
	      data_im[index_im] += data_col[index_col];     
	    }
	  }    
	}
      }
    }
  }  
}


void waffe2_col2im_s_row
(
 const float* data_col,
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
 float* data_im) {
  for (uint64_t i=0;i<N * C * H * W;i++) { data_im[i] = 0.0; } 
  uint64_t stride_imW = C * H * W;
  uint64_t stride_imH = H * W;
  uint64_t stride_imC = W;
  uint64_t stride_imN = 1;

  // COL1 = (N C K-H K-W H-out W-out)
  uint64_t stride_col6 = C * K_H * K_W * H_out * W_out;
  uint64_t stride_col5 = K_H * K_W * H_out * W_out;
  uint64_t stride_col4 = K_W * H_out * W_out;
  uint64_t stride_col3 = H_out * W_out;
  uint64_t stride_col2 = W_out;
  uint64_t stride_col1 = 1;
  
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
	      uint64_t index_col = stride_col1 * batch_n + stride_col2 * cth + stride_col3 * h_start + stride_col4 * w_start + stride_col5 * H_out_col + stride_col6 * W_out_col;
	      uint64_t index_im = stride_imN * batch_n + stride_imC * cth + stride_imH * H_out_im + stride_imW * W_out_im;	      
	      data_im[index_im] += data_col[index_col];	            
	    }
	  }    
	}
      }
    }
  }  
}

void waffe2_col2im_i_row
(
 const int* data_col,
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
 int* data_im) {
  for (uint64_t i=0;i<N * C * H * W;i++) { data_im[i] = 0; } 
  uint64_t stride_imW = C * H * W;
  uint64_t stride_imH = H * W;
  uint64_t stride_imC = W;
  uint64_t stride_imN = 1;

  // COL1 = (N C K-H K-W H-out W-out)
  uint64_t stride_col6 = C * K_H * K_W * H_out * W_out;
  uint64_t stride_col5 = K_H * K_W * H_out * W_out;
  uint64_t stride_col4 = K_W * H_out * W_out;
  uint64_t stride_col3 = H_out * W_out;
  uint64_t stride_col2 = W_out;
  uint64_t stride_col1 = 1;
  
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
	      uint64_t index_col = stride_col1 * batch_n + stride_col2 * cth + stride_col3 * h_start + stride_col4 * w_start + stride_col5 * H_out_col + stride_col6 * W_out_col;
	      uint64_t index_im = stride_imN * batch_n + stride_imC * cth + stride_imH * H_out_im + stride_imW * W_out_im;	      
	      data_im[index_im] += data_col[index_col];      
	    }
	  }    
	}
      }
    }
  }  
}
