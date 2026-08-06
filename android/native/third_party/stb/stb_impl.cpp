// Single implementation translation unit for the vendored public-domain stb
// image helpers.  Production code only includes the headers; the definitions
// live here so multiple targets never collide on duplicate symbols.
//
// The ncnn mirror of these headers omits <math.h> (upstream stb includes it
// for ldexp/pow/frexp inside the HDR/linear guards), so it is pulled in here
// explicitly.
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"