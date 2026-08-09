#include "phase18d_seam_metrics.h"

#include <iostream>
#include <string>
#include <vector>

namespace p18d = seededge::phase18d;
namespace { int failed=0; void expect(bool v,const char*m){if(!v){++failed;std::cerr<<"FAIL "<<m<<'\n';}} }

int main()
{
    p18d::TilePlan plan; plan.image_width=8; plan.image_height=4;
    p18d::TileSpec left; left.tile_index=0; left.source_left=0; left.source_top=0;
    left.source_right=5; left.source_bottom=4; left.valid_width=5; left.valid_height=4;
    p18d::TileSpec right=left; right.tile_index=1; right.source_left=4; right.source_right=8;
    right.valid_width=4;
    plan.tiles={left,right};
    std::vector<uint8_t> rgb(8*4*3);
    for(int y=0;y<4;++y) for(int x=0;x<8;++x) for(int c=0;c<3;++c)
        rgb[(y*8+x)*3+c]=static_cast<uint8_t>(x + (x>=4?100:0));
    p18d::SeamMetrics metrics; std::string error;
    expect(p18d::analyze_seams_rgb8(rgb.data(),rgb.size(),8,4,plan,&metrics,&error),
           "valid seam field analyzes");
    expect(metrics.bounds_valid && metrics.duplicate_bounds==0 && metrics.vertical_samples==4,
           "bounds and vertical sample count are exact");
    expect(metrics.vertical_mean_gradient==101.0 && metrics.vertical_p95_gradient==101.0 &&
           metrics.vertical_max_gradient==101.0,
           "injected vertical jump is measured exactly");
    expect(metrics.vertical_reference_mean_gradient==1.0,
           "neighboring non-seam gradient remains the smooth baseline");
    expect(metrics.horizontal_samples==0,
           "no horizontal boundary reports no synthetic samples");

    plan.tiles.push_back(right);
    expect(!p18d::analyze_seams_rgb8(rgb.data(),rgb.size(),8,4,plan,&metrics,&error) &&
           error=="DUPLICATE_TILE_BOUNDS", "duplicated tile bounds are rejected");
    plan.tiles.pop_back(); plan.tiles[1].source_right=9;
    expect(!p18d::analyze_seams_rgb8(rgb.data(),rgb.size(),8,4,plan,&metrics,&error) &&
           error=="INVALID_TILE_BOUNDS", "out-of-range tile crop is rejected");
    std::cout<<"phase18d_seam_metrics_test failed="<<failed<<'\n'; return failed?1:0;
}
