#include<stdio.h>
#include<math.h>
int main()
{
    int a,b,sum;
	
	printf("\n Enter two integers separated by a space; I'll tell you their sum: ");
    scanf("%d %d", &a,&b);
    sum=a+b;
	printf("\n Their sum is %d",sum);
	int quotient,remainder;
	quotient=a/b;
	remainder=a%b;
	printf("\n The quotient is %d and the remainder is %d",quotient,abs(remainder));
	// Add two more int variables called quotient and remainder;
	/* Compute the quotient and remainder when a is divided by b and store
	them in the respective variables; then print them. */
	// Input a negative integer for a; what happens?	
	
    return 0;
}
