#include "guarded_interval.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc,char** argv){
 if(argc!=4)return 2;FILE* in=fopen(argv[1],"rb");FILE* out=fopen(argv[2],"wb");if(!in||!out)return 3;
 int tasks=atoi(argv[3]);
 for(int k=0;k<tasks;k++){
  int op,n;double t;if(fread(&op,4,1,in)!=1||fread(&n,4,1,in)!=1||fread(&t,8,1,in)!=1)return 4;
  double s=0;bool okay=true;gi::I value;
  if(op<5){
   if(op==0)value=gi::log(gi::pt(t));else if(op==1)value=gi::exp(gi::pt(t));
   else if(op==2)value=gi::log1p(gi::pt(t));else if(op==3)value=gi::expm1(gi::pt(t));
   else value=gi::add(gi::pt(1.),gi::pt(t));
  }else{
   std::vector<gi::Layer> data(n);for(int j=0;j<n;j++){
    double p[7];if(fread(p,8,7,in)!=7)return 5;
    data[j]={{p[0],p[1]},{p[2],p[3]},{p[4],p[5]},p[6]};
   }
   value=gi::residual(data.data(),n,t,s,okay);
  }
  double result[4]={value.lo,value.hi,s,okay?1.:0.};if(fwrite(result,8,4,out)!=4)return 6;
 }
 fclose(in);fclose(out);return 0;
}
