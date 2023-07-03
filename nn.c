#include "./nn.h"

void testLibrary(){
    // Matrix a = matrix_alloc(2, 2);
    // Matrix b = matrix_alloc(2, 2);
    // Matrix c = matrix_alloc(2, 1);
    // Matrix destination = matrix_alloc(2, 2);
    // Matrix theMatrix = matrix_alloc(2, 2);
    // // matrix_random_init(m, 0, 10);
    // matrix_fill(a, 2);
    // matrix_fill(b, 1);
    // matrix_random_init(c, 0, 10);
    // MATRIX_AT(b, 0, 1) = 0;
    // MATRIX_AT(b, 1, 0) = 0;
    // matrix_add(destination, a, b);
    // matrix_multiply(theMatrix, a, b);
    // matrix_multiply(c, b, c);
    // matrix_print(a);
    // matrix_print(b);
    // printf("<---------------------------------------------->\n");
    // matrix_print(c);
    // matrix_print(destination);
    // matrix_print(theMatrix);

    
    double fullInput[12] = {0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0};

    Matrix allMight = matrix_alloc(4, 3);
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 3; j++){
            MATRIX_AT(allMight, i, j) = fullInput[i*3 + j];
        }
    }

    MATRIX_PRINT(allMight);

    Matrix input = matrix_sub_matrix(allMight, 0, 3, 0, 1);
    Matrix output = matrix_sub_matrix(allMight, 0, 3, 2, 2);

    MATRIX_PRINT(input);
    MATRIX_PRINT(output);

    
    int architecture[3] = {2, 2, 1};
    char* activators[3];
    char* none = "none";
    char* sgm = "sigmoid";
    activators[0] = none;
    activators[1] = sgm;
    activators[2] = sgm;
    NeuralNetwork nn = nn_alloc(architecture, 3, activators);

    nn_random_init_all_weights(nn, 0, 1);
    // printf("dats: %d\n", nn.layers[2].activations->rows);
    double initCost = nn_compute_cost(nn, input, output);
    NN_PRINT(nn);
    printf("<--------------------------------------------------->\n");
    train(nn, input, output, 50000);
    NN_PRINT(nn);
    printf("init cost: %lf \n", initCost);
    free(allMight.data);
    return;
}

int main(){
    testLibrary(); 
    return 0;
}