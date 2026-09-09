#include<stdio.h>

int main(void)
{
    double a,b,c;
	
	printf("Enter the values of a,b,c: ");
	scanf("%lf %lf %lf",&a,&b,&c);	
	
	if (((a>0)&& (b>0)&&(c>0)&&(a+b>c)&&(b+c>a)&&(c+a>b)))
	{
		printf("\n The three values represent the sides of a triangle.");				
	}
	else
	{
		printf("There cannot be a triangle with these lengths.");
	}
	
	/* Remove the ! operator and replace the logical ORs by logical ANDs
	by appropriately changing each of the 6 conditions. */ 
	
    return 0;
}
