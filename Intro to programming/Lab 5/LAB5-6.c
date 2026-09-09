#include<stdio.h>

int main(void){
    int n,temp;
    int found=0;
    scanf("%d",&n);
    int arr[n];
    for(int i=0;i<n;i++){
        scanf("%d",&temp);
        arr[i]=temp;
    }
    for(int i=0;i<n;i++){
        for(int j=i+1;j<n;j++){
            if(arr[i]==arr[j]){
                printf("Repeated Value: %d\n",arr[i]);
                printf("Positions : %d , %d",i,j);
                found=1;
                break;
            }
        }
        if(found){
            break;
        }
    }
    if(found==0){
        printf("No Repeated numbers");
    }
}