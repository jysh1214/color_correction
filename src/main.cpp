#include "color_correction.h"

int main(int argc, char** argv)
{
    if (argc < 2) {
        return -1;
    }

    cv::Mat src, dst;
    src = cv::imread(argv[1]);
    if (src.empty()) {
        return -1;
    }

    colorCorrection(src, dst);
    cv::imwrite("output.png", dst);

    return 0;
}
