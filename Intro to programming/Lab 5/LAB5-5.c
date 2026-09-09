#include<stdio.h>

int main(void){
    int n;
    float sum=0,avg,temp;
    int cnt=0;
    scanf("%d",&n);
    float arr[n];
    for(int i=0;i<n;i++){
        scanf("%f",&temp);
        arr[i]=temp;
        sum+=temp;
    }
    avg=sum/n;
    for(int i=0;i<n;i++){
        if(arr[i]>avg){
            cnt++;
        }
    }
    printf("Average: %.1f\nNumber of values above the average: %d",avg,cnt);
    
}