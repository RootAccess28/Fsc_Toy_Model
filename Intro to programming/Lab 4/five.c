#include<stdio.h>
int main(void){
    int c,u;
    c=10;
    printf("\nGuess a number between 1 and 10 :");
    while(1){

        scanf("%d",&u);
        if(c==u){
            printf("You got it!");
            break;
        }else{
            printf("Wrong answer! Try again :");
            continue;
        }

    }
}