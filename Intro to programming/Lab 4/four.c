#include<stdio.h>
#include <limits.h>

int main(void){
    int n,m,c;
    m=INT_MIN;
    printf("\nEnter then number of inputs :");
    scanf("%d",&n);
    for(int i=1;i<=n;i++){
        printf("Enter Input %d:",i);
        scanf("%d",&c);
        if(c>m) m=c;
    }
    printf("The max of the given inputs is: %d", m);

    return 0;
}