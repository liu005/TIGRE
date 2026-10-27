#include "mex.h"
#include "../../../Common/CUDA/Siddon_surface_projection.hpp"
#include <cmath>
#include <climits>
// Packed internal interface for FDK_surface; all coordinates are world xyz.
void mexFunction(int nlhs,mxArray** out,int nrhs,const mxArray** in) {
    if(nrhs!=9 || nlhs!=1) mexErrMsgIdAndTxt("TIGRE:surface:arguments","Expected nine inputs and one output.");
    for(int k=0;k<9;k++) {
        if(!mxIsSingle(in[k]) || mxIsComplex(in[k]) || mxIsSparse(in[k]))
            mexErrMsgIdAndTxt("TIGRE:surface:type","Expected real full single arrays.");
        const float* a=static_cast<const float*>(mxGetData(in[k]));
        for(mwSize j=0;j<mxGetNumberOfElements(in[k]);j++) if(!std::isfinite(a[j]))
            mexErrMsgIdAndTxt("TIGRE:surface:finite","Inputs must be finite.");
    }
    auto p=[&](int k){return static_cast<const float*>(mxGetData(in[k]));};
    for(int k=1;k<=3;k++) if(mxGetNumberOfElements(in[k])!=3)
        mexErrMsgIdAndTxt("TIGRE:surface:geometry","Expected three volume components.");
    const float *n=p(1),*d=p(2),*o=p(3);
    for(int j=0;j<3;j++) if(n[j]<1 || n[j]>=INT_MAX || n[j]!=std::floor(n[j]) || d[j]<=0)
        mexErrMsgIdAndTxt("TIGRE:surface:geometry","Invalid volume dimensions or spacing.");
    mwSize cols=mxGetNumberOfElements(in[5]),rows=mxGetNumberOfElements(in[7]),views=mxGetN(in[4]);
    if(cols<2 || !rows || !views || cols>INT_MAX || rows>INT_MAX || views>INT_MAX ||
        mxGetM(in[4])!=3 || mxGetNumberOfElements(in[6])!=cols ||
        mxGetNumberOfElements(in[0])!=cols*rows*views || mxGetNumberOfElements(in[8])!=1 ||
        p(8)[0]<0 || p(8)[0]>=INT_MAX || p(8)[0]!=std::floor(p(8)[0]))
        mexErrMsgIdAndTxt("TIGRE:surface:shape","Invalid FDK table/data dimensions or device.");
    for(mwSize j=0;j<cols;j++) if(p(6)[j]<=0 || (j && p(5)[j]<=p(5)[j-1]))
        mexErrMsgIdAndTxt("TIGRE:surface:table","Fan angles must ascend and distances be positive.");
    for(mwSize j=1;j<rows;j++) if(p(7)[j]<=p(7)[j-1])
        mexErrMsgIdAndTxt("TIGRE:surface:table","Row heights must ascend.");
    mwSize dims[3]={mwSize(n[0]),mwSize(n[1]),mwSize(n[2])};
    out[0]=mxCreateNumericArray(3,dims,mxSINGLE_CLASS,mxREAL);
    int status=fdk_surface_backprojection(p(0),int(n[0]),int(n[1]),int(n[2]),
        d[0],d[1],d[2],o[0],o[1],o[2],int(views),int(rows),int(cols),
        p(4),p(5),p(6),p(7),static_cast<float*>(mxGetData(out[0])),int(p(8)[0]));
    if(status) mexErrMsgIdAndTxt("TIGRE:surface:CUDA","Surface FDK CUDA operation failed.");
}
