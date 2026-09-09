#include<stdio.h>

int main(void)
{
    double a,b,c;
	
	printf("Enter the values of a,b,c: ");
	scanf("%lf %lf %lf",&a,&b,&c);	
	
	if ((a<=0)||(b<=0)||(c<=0))
	{
		printf("lenghts cannot be 0 or negative.");
	}
	else
	{
		if((a+b<=c)||(b+c<=a)||(c+a<=b)){
			printf("lengths violate the triangle inequality.");
		}else{
			printf("\n The three values represent the sides of a triangle.");
		}		
	}
	
	/* Rewrite the above code using nested if...else... as follows.
	If one of the values is negative, print that lengths cannot be negative.
	Else do the following:
		Check if the triangle inequality is violated. If yes,
		print that the lengths violate the triangle inequality.
		If not, print that the values represent the sides of a triangle.
	*/
	
    return 0;
}
