#include<stdio.h>

int main(void){
    int n;
    scanf("%d",&n);
    float temp;
    float a1[n],a2[n];
    float dp=0;
    for(int i=0;i<n;i++){
        scanf("%f",&temp);
        a1[i]=temp;
    }
    for(int i=0;i<n;i++){
        scanf("%f",&temp);
        a2[i]=temp;
    }
    for(int i=0;i<n;i++){
        dp+= a1[i]*a2[i];
    }
    printf("%f",dp);
    
}