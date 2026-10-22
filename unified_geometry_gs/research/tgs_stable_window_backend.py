"""Isolated v2 local-coordinate forward, with direct/reconstructed radius VJP."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from tgs_native_diagnostic_backend import ROOT,SOURCE
from tgs_backward_fixed_backend import once

OUT=ROOT/'experiments/tgs_stable_window_v2'


def forward_transform(text):
    begin=text.index('\t float4 p_hom_center =')
    end=text.index('\trect_max[idx] =',begin)
    text=text[:begin]+r'''
    // Stable projection from the original float32 world/matrix inputs.
    double2 q64[3];
    for (int i=0; i<3; ++i) {
        const double x=triangles_points[offset+3*i], y=triangles_points[offset+3*i+1], z=triangles_points[offset+3*i+2];
        const double hx=(double)projmatrix[0]*x+(double)projmatrix[4]*y+(double)projmatrix[8]*z+projmatrix[12];
        const double hy=(double)projmatrix[1]*x+(double)projmatrix[5]*y+(double)projmatrix[9]*z+projmatrix[13];
        const double hw=(double)projmatrix[3]*x+(double)projmatrix[7]*y+(double)projmatrix[11]*z+projmatrix[15];
        const double inv=1.0/(hw+1e-7);
        q64[i]=make_double2((hx*inv+1.0)*(0.5*W)-0.5,(hy*inv+1.0)*(0.5*H)-0.5);
        p_w[cumsum_for_triangle+i]=(float)inv;
    }
    // An integer anchor keeps exact integer pixel subtraction at tested sizes.
    const double ax=floor((q64[0].x+q64[1].x+q64[2].x)/3.0);
    const double ay=floor((q64[0].y+q64[1].y+q64[2].y)/3.0);
    const float2 center_triangle_2D=make_float2((float)ax,(float)ay);
    float distance_points=0.0f;
    for (int i=0; i<3; ++i) {
        p_image[cumsum_for_triangle+i]=make_float2((float)(q64[i].x-ax),(float)(q64[i].y-ay));
    }
    const float2 A1=p_image[cumsum_for_triangle], B1=p_image[cumsum_for_triangle+1], C1=p_image[cumsum_for_triangle+2];
    const float a=hypotf(B1.x-C1.x,B1.y-C1.y), b=hypotf(A1.x-C1.x,A1.y-C1.y), c=hypotf(A1.x-B1.x,A1.y-B1.y);
    const float sum=a+b+c;
    const float twice_area=(B1.x-A1.x)*(C1.y-A1.y)-(B1.y-A1.y)*(C1.x-A1.x);
    const float radius=fabsf(twice_area)/sum;
    const float2 incenter=make_float2((a*A1.x+b*B1.x+c*C1.x)/sum,(a*A1.y+b*B1.y+c*C1.y)/sum);
    const float2 centroid=make_float2((A1.x+B1.x+C1.x)/3.0f,(A1.y+B1.y+C1.y)/3.0f);
    // Preserve extent relative to projected world centroid, not anchor.
    const double cx=((double)triangles_points[offset]+triangles_points[offset+3]+triangles_points[offset+6])/3.0;
    const double cy=((double)triangles_points[offset+1]+triangles_points[offset+4]+triangles_points[offset+7])/3.0;
    const double cz=((double)triangles_points[offset+2]+triangles_points[offset+5]+triangles_points[offset+8])/3.0;
    const double cw=(double)projmatrix[3]*cx+(double)projmatrix[7]*cy+(double)projmatrix[11]*cz+projmatrix[15]+1e-7;
    const double sx=(((double)projmatrix[0]*cx+(double)projmatrix[4]*cy+(double)projmatrix[8]*cz+projmatrix[12])/cw+1.0)*(0.5*W)-0.5;
    const double sy=(((double)projmatrix[1]*cx+(double)projmatrix[5]*cy+(double)projmatrix[9]*cz+projmatrix[13])/cw+1.0)*(0.5*H)-0.5;
    for(int i=0;i<3;++i) distance_points=fmaxf(distance_points,(float)hypot(q64[i].x-sx,q64[i].y-sy));
    const float dist=-radius;
    if (!isfinite(radius) || radius<1.0f || distance_points>1600.0f || distance_points<1.0f) return;
    for(int i=0;i<3;++i) {
        const float2 v=p_image[cumsum_for_triangle+i], t=p_image[cumsum_for_triangle+(i+1)%3];
        const float nx=t.y-v.y, ny=-(t.x-v.x), inv=1.0f/hypotf(nx,ny);
        float2 n=make_float2(nx*inv,ny*inv);
        float o=-(n.x*v.x+n.y*v.y);
        if(n.x*centroid.x+n.y*centroid.y+o>0) {n.x=-n.x;n.y=-n.y;o=-o;}
        normals[cumsum_for_triangle+i]=n;offsets[cumsum_for_triangle+i]=o;
    }
    // Exact real-arithmetic level set: s+(1-u0)*(P-s).
    const double shrink=1.0-pow((1.0/255.0)/(double)opacities[idx],1.0/(double)sigma[idx]);
    double minx=INFINITY,miny=INFINITY,maxx=-INFINITY,maxy=-INFINITY;
    for(int i=0;i<3;++i) {
        const float2 v=p_image[cumsum_for_triangle+i];
        const double x=ax+incenter.x+shrink*((double)v.x-incenter.x);
        const double y=ay+incenter.y+shrink*((double)v.y-incenter.y);
        minx=fmin(minx,x);maxx=fmax(maxx,x);miny=fmin(miny,y);maxy=fmax(maxy,y);
    }
    uint2 rect_min_triangle_test=make_uint2(
        (uint)fmin((double)grid.x,fmax(0.0,floor((minx-1.0)/BLOCK_X))),
        (uint)fmin((double)grid.y,fmax(0.0,floor((miny-1.0)/BLOCK_Y))));
    uint2 rect_max_triangle_test=make_uint2(
        (uint)fmin((double)grid.x,fmax(0.0,floor((maxx+1.0)/BLOCK_X)+1.0)),
        (uint)fmin((double)grid.y,fmax(0.0,floor((maxy+1.0)/BLOCK_Y)+1.0)));
    const float size=-radius*(float)(1.0-shrink);

''' +text[end:]
    text=once(text,'collected_normals[base + k].x * pixf.x','collected_normals[base + k].x * (pixf.x-collected_xy[j].x)')
    text=once(text,'collected_normals[base + k].y * pixf.y','collected_normals[base + k].y * (pixf.y-collected_xy[j].y)')
    return text


def backward_transform(text,contraction):
    text=once(text,'if (normal.x * points_xy_image[idx].x + normal.y * points_xy_image[idx].y + offset > 0)',
        'const float2 center_local=make_float2((p_image[cumsum_for_triangle].x+p_image[cumsum_for_triangle+1].x+p_image[cumsum_for_triangle+2].x)/3.0f, (p_image[cumsum_for_triangle].y+p_image[cumsum_for_triangle+1].y+p_image[cumsum_for_triangle+2].y)/3.0f);\n        if (normal.x * center_local.x + normal.y * center_local.y + offset > 0)')
    text=once(text,'collected_normals[base + k].x * pixf.x','collected_normals[base + k].x * (pixf.x-collected_xy[j].x)')
    text=once(text,'collected_normals[base + k].y * pixf.y','collected_normals[base + k].y * (pixf.y-collected_xy[j].y)')
    text=once(text,'float dL_dnormal_x = dL_dphi_x * pixf.x;','float dL_dnormal_x = dL_dphi_x * (pixf.x-collected_xy[j].x);')
    text=once(text,'float dL_dnormal_y = dL_dphi_x * pixf.y;','float dL_dnormal_y = dL_dphi_x * (pixf.y-collected_xy[j].y);')
    if contraction:
        text=once(text,'float loss_points_y[MAX_NB_POINTS] = {0.0f};','float loss_points_y[MAX_NB_POINTS] = {0.0f};\n     float log_radius_gradient=0.0f;')
        text=once(text,'// Add gradients from offset to normal gradients',
            'log_radius_gradient -= normal.x*dL_dnormal_x + normal.y*dL_dnormal_y + offset*dL_doffset;\n        // Add gradients from offset to normal gradients')
        text=once(text,'atomicAdd(&dL_dsigma_factor[global_id], -dL_dC * sigma_pre * Cx);','// Gamma reconstructed from local edge cotangents.')
        text=once(text,'dL_dsigma_factor[idx]*dlogr_x','log_radius_gradient*dlogr_x')
        text=once(text,'dL_dsigma_factor[idx]*dlogr_y','log_radius_gradient*dlogr_y')
    return text


def load_backend(variant):
    assert variant in ['direct','contraction']
    for e in json.loads((SOURCE/'manifest.json').read_text())['files']:
        assert hashlib.sha256((SOURCE/e['path']).read_bytes()).hexdigest()==e['sha256']
    previous=ROOT/'experiments/tgs_branch_aux_v1/piecewise_build/generated'
    assert hashlib.sha256((previous/'forward.cu').read_bytes()).hexdigest()=='a2c1c6a226f87d913a7b8e931e0aa797a4b015d8127cf15aad50d9ac164623be'
    assert hashlib.sha256((previous/'backward.cu').read_bytes()).hexdigest()=='1fab2a104cfc05850abd960da5f2f9503b4165240b6fc88ab7414b79e416992a'
    base=OUT/variant;generated=base/'generated';build=base/'build'
    generated.mkdir(parents=True,exist_ok=True);build.mkdir(exist_ok=True)
    for name,content in [('forward.cu',forward_transform((previous/'forward.cu').read_text())),('backward.cu',backward_transform((previous/'backward.cu').read_text(),variant=='contraction'))]:
        p=generated/name
        if p.exists():assert p.read_text()==content
        else:p.write_text(content)
    native=SOURCE/'native'
    sources=[native/'cuda_rasterizer/rasterizer_impl.cu',generated/'forward.cu',generated/'backward.cu',native/'cuda_rasterizer/utils.cu',native/'rasterize_points.cu',native/'ext.cpp']
    os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    import torch
    from torch.utils.cpp_extension import load
    started=time.time()
    mod=load(name='tgs_diag_stable_window_v2_'+variant,sources=[str(p) for p in sources],extra_include_paths=[str(native/'third_party/glm'),str(native/'cuda_rasterizer')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math'],build_directory=str(build),verbose=True)
    r=dict(module_file=mod.__file__,module_sha256=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],protocol_sha256=hashlib.sha256((ROOT/'TGS_STABLE_WINDOW_PROTOCOL_V2.md').read_bytes()).hexdigest(),seconds=time.time()-started,torch=torch.__version__,cuda=torch.version.cuda,training=False)
    receipt=base/'build_receipt.json'
    if not receipt.exists():receipt.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r),flush=True)
    return mod


if __name__=='__main__':
    for variant in ['direct','contraction']:load_backend(variant)
