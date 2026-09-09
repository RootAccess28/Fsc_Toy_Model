#include<stdio.h>
#include<math.h>

int main(void){
    int n;
    float x;
    printf("\nEnter a positive no. n:");
    scanf("%d",&n);
    for(int i=0;i<n;i++){
        printf("*");
    }
    printf("\n");
    printf("Square no. smaller than or equal to %d = ",n);
    for(int i=1;(i*i)<=n;i++){
        printf("%d ",i*i);
    }
    printf("\nEnter a floating point no. x:");
    scanf("%f",&x);
    for(int i=1;i<=n;i++){
        printf("%f ^ %d = %.3f\n",x,i,pow(x,i));
    }


    
    return 0;
}