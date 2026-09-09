#include<stdio.h>

int main(void)
{
    int n=30;
	int i=1;
    
    while (i<=n) 
    {
        if (i%3==0)
			printf("\n");
		if (i%3==1)
			printf(" Hello");
		if (i%3==2)
			printf(" Bye");
		i=i+1;
	}
	return 0;
}

/*
Trace of the variables i


*/
