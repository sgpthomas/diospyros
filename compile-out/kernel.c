#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <xtensa/sim.h>
#include <xtensa/tie/xt_pdxn.h>
#include <xtensa/tie/xt_timer.h>
#include <xtensa/xt_profiling.h>
/*
Git revision: 8451785

Git status clean
*/
int __attribute__((section(".dram0.data"))) Z[4] = {0, 0, 0, 0};
float __attribute__((section(".dram0.data"))) v_0[4] = {0.0, 0, 0, 0};
int __attribute__((section(".dram0.data"))) v_1[4] = {0, 1, 2, 3};
int __attribute__((section(".dram0.data"))) v_3[4] = {0, 0, 0, 0};
int __attribute__((section(".dram0.data"))) v_3_0[4] = {0, 0, 0, 0};
void kernel(float * a_in, float * b_out) {
float * __restrict a_in_mut = a_in;
  valign align_a_in;
  align_a_in = PDX_LA_MXF32_PP((xb_vecMxf32 *) a_in);
  float * __restrict b_out_mut = b_out;
  valign align_b_out;
  xb_vecMxf32 a_in_0_4;
  PDX_LAV_MXF32_XP(a_in_0_4, align_a_in, (xb_vecMxf32 *) a_in_mut, 16);
  xb_vecMxf32 a_in_4_8;
  PDX_LAV_MXF32_XP(a_in_4_8, align_a_in, (xb_vecMxf32 *) a_in_mut, 16);
  xb_vecMxf32 v_2;
  v_2 = a_in_0_4;
  xb_vecMxf32 v_4;
  v_4 = PDX_MOV_MXF32_FROM_MX32(PDX_SHFL_MX32(*((xb_vecMxf32 *) v_0), *((xb_vecMx32 *) v_3_0)));
  PDX_SAV_MXF32_XP(v_2, align_b_out, (xb_vecMxf32 *) b_out, 16);
  PDX_SAV_MXF32_XP(v_4, align_b_out, (xb_vecMxf32 *) b_out, 16);
  PDX_SAPOS_MXF32_FP(align_b_out, (xb_vecMxf32 *) b_out);
}