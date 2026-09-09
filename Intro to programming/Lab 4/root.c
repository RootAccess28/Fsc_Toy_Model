#include<stdio.h>

int main(void)
{
    int n=1987;
    int a=1;
    int count=0;
    for(int i=2;i*i<=n;i++){
        count++;
        if(n%i==0){
            a=0;
            break;
        }

    }
    printf("%d  %d",a,count);

	return 0;
}