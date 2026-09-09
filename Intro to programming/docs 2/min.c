#include<stdio.h>
#include<math.h>
int main(void)
{
    int a,b,c;
	
	printf("Enter the values of a,b,c: ");
	scanf("%d%d%d",&a,&b,&c);
    if(a<b){
        if(a<=c){
            printf("a is the smallest");
        }else printf("c is the smallest");
    }else{
        if(b<=c){
            printf("b is the smallest");
        }else{
            printf("c is the smallest");
        }
    }
}