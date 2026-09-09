#include<stdio.h>

int main(void)
{
    int age;
	
	printf("\n Enter your age: ");
    scanf("%d",&age);
	if ((age>=13) && (age<=19))
	{
		printf("\n You are a teenager.");
	}
	else if(age<13) printf("\n You are a child.");
	else printf("\n You are an adult.");
	/* When you combine two logical expressions, put each one inside () to avoid
	any ambiguity, as in the example above*/
    

	// Check if the age is less than 13 and if so, print that the user is a child.
	// If the age is 18 or above, print that the user is an adult.
	// Test with examples.
	
    return 0;
}
