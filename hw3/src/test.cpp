#include<iostream>
#include<vector>
using namespace std;
int total = 0;
int shape[4] = {2,3,4,5};
int strides[4]={60,20,5,1};
void nloopfor(int depth,int* out,int *a){
    int cnt=0;
    int cur = 0;
    int tab[4]={0};
    for(*tab = 0; cur >=0;){
        ++tab[cur];
        if(tab[cur]<=shape[cur]){
            if(cur==depth-1){//最后一层的循环
                int temp=0;
                for(int k=0;k<4;k++){
                    temp+=(tab[k]-1)*strides[k];
                }
                cout<<"temp:"<<temp<<endl;
                out[cnt++]=a[temp];
            }else{
            ++cur;//进入下一层
            tab[cur] = 0;//重置该层的循环次数
            }
        }else{
            --cur;//最后一层循环结束，回退一层。
        }
    }
    cout<<"total: "<<total;
}

int main(){
    int a[120];
    int out[120];
    for(int j=0;j<120;j++){
        a[j]=1;
        if (j==119)
        {
            a[j]=120;
        }
        
    }
    int n;
    nloopfor(4,out,a);
    for(int i=0;i<120;i++){
        cout<<out[i];
    }
    return 0;
}
