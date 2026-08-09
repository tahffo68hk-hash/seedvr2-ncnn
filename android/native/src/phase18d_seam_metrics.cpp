#include "phase18d_seam_metrics.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <tuple>
#include <vector>

namespace seededge::phase18d {
namespace {
bool fail(std::string* error, const char* code) { if(error)*error=code; return false; }
double gradient(const uint8_t* rgb, int width, int x0, int y0, int x1, int y1)
{
    const size_t a=(static_cast<size_t>(y0)*width+x0)*3;
    const size_t b=(static_cast<size_t>(y1)*width+x1)*3;
    return (std::abs(int(rgb[a])-int(rgb[b]))+
            std::abs(int(rgb[a+1])-int(rgb[b+1]))+
            std::abs(int(rgb[a+2])-int(rgb[b+2])))/3.0;
}
void summarize(std::vector<double> values,double*mean,double*p95,double*maximum)
{
    if(values.empty()){*mean=*p95=*maximum=0.;return;}
    *mean=std::accumulate(values.begin(),values.end(),0.)/values.size();
    std::sort(values.begin(),values.end());
    const size_t index=static_cast<size_t>(std::ceil(values.size()*0.95))-1;
    *p95=values[index];*maximum=values.back();
}
double mean(const std::vector<double>& values)
{
    return values.empty()?0.:std::accumulate(values.begin(),values.end(),0.)/values.size();
}
}

bool analyze_seams_rgb8(const uint8_t* rgb,size_t byte_count,int width,int height,
                        const TilePlan& plan,SeamMetrics* metrics,std::string* error)
{
    if(error)error->clear(); if(!metrics)return fail(error,"SEAM_METRICS_OUTPUT_NULL");
    *metrics=SeamMetrics{};
    if(!rgb||width<=0||height<=0||byte_count!=static_cast<size_t>(width)*height*3)
        return fail(error,"INVALID_RGB_BUFFER");
    std::set<std::tuple<int,int,int,int>> bounds;
    std::set<int> vertical,horizontal;
    for(const TileSpec& tile:plan.tiles){
        if(tile.source_left<0||tile.source_top<0||tile.source_right>width||
           tile.source_bottom>height||tile.source_left>=tile.source_right||
           tile.source_top>=tile.source_bottom||
           tile.valid_width!=tile.source_right-tile.source_left||
           tile.valid_height!=tile.source_bottom-tile.source_top)
            return fail(error,"INVALID_TILE_BOUNDS");
        const auto key=std::make_tuple(tile.source_left,tile.source_top,tile.source_right,tile.source_bottom);
        if(!bounds.insert(key).second){metrics->duplicate_bounds++;return fail(error,"DUPLICATE_TILE_BOUNDS");}
        if(tile.source_left>0)vertical.insert(tile.source_left);
        if(tile.source_top>0)horizontal.insert(tile.source_top);
    }
    std::vector<double> vg,hg,vr,hr;
    for(int x:vertical)for(int y=0;y<height;++y){
        vg.push_back(gradient(rgb,width,x-1,y,x,y));
        if(x>=2)vr.push_back(gradient(rgb,width,x-2,y,x-1,y));
        if(x+1<width)vr.push_back(gradient(rgb,width,x,y,x+1,y));
    }
    for(int y:horizontal)for(int x=0;x<width;++x){
        hg.push_back(gradient(rgb,width,x,y-1,x,y));
        if(y>=2)hr.push_back(gradient(rgb,width,x,y-2,x,y-1));
        if(y+1<height)hr.push_back(gradient(rgb,width,x,y,x,y+1));
    }
    metrics->bounds_valid=true;metrics->vertical_samples=vg.size();metrics->horizontal_samples=hg.size();
    summarize(vg,&metrics->vertical_mean_gradient,&metrics->vertical_p95_gradient,&metrics->vertical_max_gradient);
    summarize(hg,&metrics->horizontal_mean_gradient,&metrics->horizontal_p95_gradient,&metrics->horizontal_max_gradient);
    metrics->vertical_reference_mean_gradient=mean(vr);metrics->horizontal_reference_mean_gradient=mean(hr);
    return true;
}

}  // namespace seededge::phase18d
