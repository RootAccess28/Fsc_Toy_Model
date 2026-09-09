#include<stdio.h>

int main(void)
{
    int n=1000;
	int count=0;
    
    while (n>0) 
    {
          //printf("\n %d",n);
		  n=n/2;
		  count=count+1;
	}
	printf("\n The number of iterations was %d",count);
	return 0;
}

// How many iterations as a function of n does the above loop take?

/*
Trace of the variable n
n: 100
n: 50
n: 25
n: 12
n: 6
n: 3
n: 1
n: 0

*/
