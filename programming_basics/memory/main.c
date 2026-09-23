#include <stdio.h>
#include <stdlib.h>

int main() {
    // Allocate memory for an array of 5 integers
    int *arr = (int *)malloc(5 * sizeof(int));

    if (arr == NULL) {
        // Check if allocation was successful
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Set values in the allocated memory
    for (int i = 0; i < 5; i++) {
        arr[i] = i * 2;  // Set the values
    }

    // Print the values
    for (int i = 0; i < 5; i++) {
        printf("arr[%d] = %d\n", i, arr[i]);
    }

    // Free the allocated memory
    free(arr);

    return 0;
}

