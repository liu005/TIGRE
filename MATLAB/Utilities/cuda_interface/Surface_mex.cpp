#include "mex.h"
#include "../../../Common/CUDA/Siddon_surface_projection.hpp"
#include <cmath>
#include <climits>
#include <initializer_list>

// Internal packed interface. Public callers use Ax_surface / Atb_surface.
void mexFunction(int nlhs,mxArray** out,int nrhs,const mxArray** in) {
    if(nrhs!=9 || nlhs!=1) mexErrMsgIdAndTxt("TIGRE:surface:arguments","Expected nine inputs and one output.");
    for(int k=0;k<9;k++)
        if(mxIsComplex(in[k]) || mxIsSparse(in[k]) ||
           (k==7 ? !mxIsUint8(in[k]) : !mxIsSingle(in[k])))
            mexErrMsgIdAndTxt("TIGRE:surface:type","Expected real full single arrays (mask uint8).");
    if(mxGetNumberOfElements(in[0])!=1 || mxGetNumberOfElements(in[8])!=1)
        mexErrMsgIdAndTxt("TIGRE:surface:scalar","Mode and device must be scalar.");
    for(int k=2;k<=4;k++) if(mxGetNumberOfElements(in[k])!=3)
        mexErrMsgIdAndTxt("TIGRE:surface:geometry","Volume vectors need three entries.");
    auto ptr=[&](int k){return static_cast<const float*>(mxGetData(in[k]));};
    const float *n=ptr(2), *d=ptr(3), *o=ptr(4);
    for(int k=0;k<3;k++) if(!std::isfinite(n[k]) || n[k]<1 || n[k]>=INT_MAX || n[k]!=std::floor(n[k]) ||
        !std::isfinite(d[k]) || d[k]<=0 || !std::isfinite(o[k]))
        mexErrMsgIdAndTxt("TIGRE:surface:geometry","Invalid volume geometry.");
    bool back=ptr(0)[0]==1;
    if((ptr(0)[0]!=0 && !back) || !std::isfinite(ptr(8)[0]) || ptr(8)[0]<0 || ptr(8)[0]>=INT_MAX || ptr(8)[0]!=std::floor(ptr(8)[0]))
        mexErrMsgIdAndTxt("TIGRE:surface:mode","Invalid mode or zero-based device.");
    const mwSize* sz=mxGetDimensions(in[6]);
    mwSize nd=mxGetNumberOfDimensions(in[6]);
    if(nd>4 || sz[0]!=3) mexErrMsgIdAndTxt("TIGRE:surface:pixels","Packed pixels must be 3 x columns x rows x views.");
    mwSize cols=sz[1], rows=nd>2?sz[2]:1, pv=nd>3?sz[3]:1;
    mwSize views=mxGetN(in[5]);
    if(mxGetM(in[5])!=3 || !views || (pv!=1 && pv!=views) || rows>INT_MAX || cols>INT_MAX || views>INT_MAX)
        mexErrMsgIdAndTxt("TIGRE:surface:views","Invalid source/pixel view dimensions.");
    mwSize plane=rows*cols, mv=mxGetNumberOfElements(in[7])/plane;
    if(mxGetNumberOfElements(in[7])!=mv*plane || (mv!=1 && mv!=views))
        mexErrMsgIdAndTxt("TIGRE:surface:mask","Invalid packed mask size.");
    mwSize vox=mwSize(n[0])*mwSize(n[1])*mwSize(n[2]);
    if(mxGetNumberOfElements(in[1])!=(back?plane*views:vox))
        mexErrMsgIdAndTxt("TIGRE:surface:data","Input data size mismatch.");
    for(int k: {1,5,6}) for(mwSize j=0;j<mxGetNumberOfElements(in[k]);j++)
        if(!std::isfinite(ptr(k)[j])) mexErrMsgIdAndTxt("TIGRE:surface:finite","Data and coordinates must be finite.");
    mwSize dims[3]={back?mwSize(n[0]):cols,back?mwSize(n[1]):rows,back?mwSize(n[2]):views};
    out[0]=mxCreateNumericArray(3,dims,mxSINGLE_CLASS,mxREAL);
    auto fn=back?siddon_surface_backprojection:siddon_surface_projection;
    int status=fn(ptr(1),int(n[0]),int(n[1]),int(n[2]),d[0],d[1],d[2],o[0],o[1],o[2],
        int(views),int(rows),int(cols),ptr(5),ptr(6),int(pv),
        static_cast<const unsigned char*>(mxGetData(in[7])),int(mv),
        static_cast<float*>(mxGetData(out[0])),int(ptr(8)[0]));
    if(status) mexErrMsgIdAndTxt("TIGRE:surface:CUDA","Surface CUDA operation failed; see CUDA diagnostic above.");
}
