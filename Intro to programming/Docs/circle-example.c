#include<stdio.h>
#define PI 3.1416
int main()
{
	double radius, area;
	
	printf("\n Enter the radius of a circle; I'll tell you its area and circumference: ");
	scanf("%lf",&radius);
	area=PI*radius*radius;
	printf("The area is: %4.4f",area);
	// Change the 4.4 to 4.2 or 4.6. What happens? Remember to recompile.
	/* Can you guess how the number before f formats 
	the area variable during print?7*/
	double circumference;
	circumference=2*PI*radius;
	/*Create a new variable called circumference.
	Compute its value for the given radius and print it.*/
	printf("\nThe circumference is: %4.4f",circumference);
    return 0;
}
