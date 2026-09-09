#include<stdio.h>
#include<math.h>
int main(void)
{
    char ann[1];
    printf("Who is the Cr\n a. me \n b. you \n c. nihant \n d. none of the above \n");
    scanf("%s", ann);
    printf("You entered: %s", ann);
    if(ann[0]=='c') printf("\nCorrect answer");
    else printf("\nWrong answer");
}