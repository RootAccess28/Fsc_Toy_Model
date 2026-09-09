#include<stdio.h>
#include<math.h>

int main(void)
{
    int n=12345;
	int count=0;
    int len;
    len=(int)log10(n)-1;
    // while (n>0) 
    // {
    //       //printf("\n %d",n);
	// 	  n=n/10;
	// 	  count=count+1;
	// }

    while(n>0){
        if(len==0){
            printf("%d",n);
            break;
        }
        
        int crr=n/pow(10,len);
        printf("%d",crr);
        int num=(pow(10,len));
        n=n%num;
        len--;
    }
	return 0;
}