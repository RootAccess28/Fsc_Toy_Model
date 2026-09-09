#include<stdio.h>

int main(void){
    int r,n;
    float p;
    p=1.0;
    n=0;
    printf("\nEnter the rate of interest in percent:");
    scanf("%d",&r);
    while(p<2.0){
        p*= (100.0+r)/100.0;
        n+=1;
    }
    printf("No. of years required to atleast double the principal amount is : %d",n);


    return 0;
}