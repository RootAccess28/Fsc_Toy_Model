#include<stdio.h>

int main(void)
{
	int n;
	printf("Enter an integer: ");
	scanf("%d",&n);
	
	if (n%2==0)
	{
		printf("The number is even.");
	}
	else
	{
		printf("The number is odd.");
	}
	
	/* Put if...else statements into blocks, i.e. inside {} and 
	indent, i.e. use a tab or spaces for the statements inside a block
	for better readability */
	
	return 0;
}