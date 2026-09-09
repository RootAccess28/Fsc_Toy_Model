#include<stdio.h>

int main(void){
    int n,temp;
    scanf("%d",&n);
    int hash[10]={0};
    for(int i=0;i<n;i++){
        scanf("%d",&temp);
        hash[temp]++;
    }
    for(int i=0;i<10;i++){
        int crr=hash[i];
        printf("%d: %d\n",i,crr);
    }
}