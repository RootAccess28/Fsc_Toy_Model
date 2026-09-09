#include<stdio.h>

int main(void){
    int m,n;
    scanf("%d",&m);
    scanf("%d",&n);
    for(int i=1;i<=m;i++){
        for(int j=1;j<=n;j++){
            if((i==1) || (i==m)|| (j==1) || (j==n)){
                printf("* ");
            }else{
                printf("  ");
            }
        }
        printf("\n");
    }
    
    
}