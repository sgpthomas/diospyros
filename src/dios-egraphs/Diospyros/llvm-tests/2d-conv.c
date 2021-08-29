#include <stdio.h>

#define I_ROWS 2
#define I_COLS 2
#define F_ROWS 2
#define F_COLS 2
#define O_ROWS ((I_ROWS + F_ROWS) - 1)
#define O_COLS ((I_COLS + F_COLS) - 1)

void convolution(float mat_in[I_ROWS * I_COLS], float f_in[F_ROWS * F_COLS],
                 float mat_out[O_ROWS * O_COLS]) {
    for (int outRow = 0; outRow < O_ROWS; outRow++) {
        for (int outCol = 0; outCol < O_COLS; outCol++) {
            for (int fRow = 0; fRow < F_ROWS; fRow++) {
                for (int fCol = 0; fCol < F_COLS; fCol++) {
                    int fRowTrans = F_ROWS - 1 - fRow;
                    int fColTrans = F_COLS - 1 - fCol;
                    int iRow = outRow - fRowTrans;
                    int iCol = outCol - fColTrans;

                    if (iRow >= 0 && iRow < I_ROWS && iCol >= 0 &&
                        iCol < I_COLS) {
                        // float v = mat_in[iRow * I_COLS + iCol] *
                        //           f_in[fRowTrans * F_COLS + fColTrans];
                        // mat_out[outRow * O_COLS + outCol] += v;

                        mat_out[outRow * O_COLS + outCol] +=
                            mat_in[iRow * I_COLS + iCol] *
                            f_in[fRowTrans * F_COLS + fColTrans];
                    }
                }
            }
        }
    }
}

int main(void) {
    // Example due to Song Ho Anh.
    // http://www.songho.ca/dsp/convolution/convolution2d_example.html
    float mat_in[I_ROWS * I_COLS] = {1, 2, 3, 4};
    // {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    float f_in[F_ROWS * F_COLS] = {1, 1, 1, 1};
    float mat_out[O_ROWS * O_COLS] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    // = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    convolution(mat_in, f_in, mat_out);
    for (int i = 0; i < O_ROWS * O_COLS; i++) {
        printf("output: %f\n", mat_out[i]);
    }
    // printf("first: %f\n", mat_out[0][0]);
    // printf("second: %f\n", mat_out[0][1]);
    // printf("third: %f\n", mat_out[0][2]);
    // printf("fourth: %f\n", c_out[1][1]);
    // expected (-13, -20, -17)
    return 0;
}