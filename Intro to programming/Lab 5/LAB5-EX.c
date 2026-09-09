#include<stdio.h>

int main(void){
    int n;
    scanf("%d",&n);
    float temp;
    float a[n],b[n];
    for(int i=0;i<n;i++){
        scanf("%f",&temp);
        a[i]=temp;
    }
    for(int i=0;i<n;i++){
        scanf("%f",&temp);
        b[i]=temp;
    }
    float res=0;
    for(int j=n-1;j>=0;j--){
        if(j==n-1){
            res+=a[j]+b[j];
        }else{
            res= a[j]+ (res*b[j])/(res+b[j]);
        }
    }
    printf("%f",res);
}