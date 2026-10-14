#include <stdio.h>

int main() {
    int m, n, T;
    int image[100][100];

    scanf("%d %d", &m, &n);
    scanf("%d", &T);

    /* Read the grayscale image */
    for(int x = 0; x<m; x++) {
        for(int y=0; y<n; y++) {
            scanf("%d", &image[x, y]);
        }
    }

    /* Convert it to a black-and-white image */
    for(int x = 0; x<m; x++) {
        for(int y=0; y<n; y++) {
            if(image[x][y]>=T) {
                image[x][y]=255;
            }
            else {
                image[x][y]=0;
            }
        }
    }
    /* Print the resulting image */
    for(int x = 0; x<m; x++) {
        for(int y=0; y<n; y++) {
            printf("%d ", image[x][y]);
        }
        printf("\n");
    }
    return 0;
}