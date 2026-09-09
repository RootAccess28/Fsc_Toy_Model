#include<stdio.h>

int main(void)
{
    int a=1,b;
	
	printf("Enter the value of b: ");
	scanf("%d",&b);	
    
    while (a<b) 
    {
		printf("\n %d %d",a,b);
        a=a+1;
		b=b-1;
	}
	printf("\n %d %d",a,b);
	return 0;
}

/*
Trace of the variables a,b,difference
Assume b is even.
1,b, b-1
2,b-1, b-3
3,b-2, b-5

r,r+1, 1

b/2

*/
