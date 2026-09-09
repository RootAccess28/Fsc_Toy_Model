#include<stdio.h>
#include<math.h>
int main(void)
{
    int a,b,c;
	
	printf("Enter the values of a,b,c: ");
	scanf("%d%d%d",&a,&b,&c);
    int disc= (int)(pow(b,2) - 4*a*c);
    if(disc==0) printf("Quadratic eqn has equal real roots");
    else if(disc<0) printf("Quadratic eqn has no real roots");
    else printf("Quadratic eqn has distinct real roots");
}