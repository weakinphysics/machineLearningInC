#ifndef NN_H
#define NN_H
#include <stddef.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <pthread.h>
#include <string.h>


//defines NN_MALLOC to use the stdlib malloc(memory allocator). You can define your own memory allocator and name it NN_MALLOC to use on your machine as per requirements

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif

//same as NN_MALLOC you can define your own custom assert in the c files that include this header file

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif

typedef struct{
    //this creates a struct that has a data pointer to a linear chunk of memory
    //of size rows x cols 
    int rows;
    int cols;
    double maxi;
    double mini;
    int stride;
    double *data;    
} Matrix;

typedef struct{
    double leakage;
    char* activation;
    Matrix weights;
    Matrix biases;
    Matrix activations;
}Layer;

typedef struct{
    double epsilon;
    double alpha;
    char** activationFunctions;
    int layerCount;
    int *architecture;
    Layer *layers;

}NeuralNetwork;



#define MATRIX_PRINT(m) matrix_print(m, #m)
#define NN_PRINT(nn) nn_print(nn, #nn)
#define MATRIX_AT(m, i, j) m.data[(i)*((m).stride) + (j)]
#define SGM(x) sigmoid((x))
#define ARRAY_LEN(y) sizeof((y))/sizeof((y)[0])
#define NN_LOAD_INPUT(ann, z) matrix_copy((ann).activations[0], z);
double generate_randoms(double, double);
double sigmoid(double);
Matrix matrix_alloc(int, int);
Matrix matrix_sub_matrix(Matrix, int, int, int, int);
void matrix_print(Matrix, char*);
void matrix_print_nn(Matrix, char*, int);
void matrix_random_init(Matrix, double, double);
void matrix_fill(Matrix , double);
void matrix_multiply(Matrix, Matrix, Matrix);
void matrix_add(Matrix, Matrix, Matrix);
void matrix_copy(Matrix, Matrix);
Matrix matrix_get_row(Matrix, int);
NeuralNetwork nn_alloc(int *, int, char**);
void nn_print(NeuralNetwork, char*);
void nn_forward(NeuralNetwork, Matrix, Matrix);
double cost(NeuralNetwork, Matrix, Matrix);

#endif



#ifndef NN_IMPLEMENTATION
#define NN_IMPLEMENTATION

double generate_randoms(double maxi, double mini){
    double randMaxDouble = RAND_MAX*(2.00 - 1.00);
    double randomNumber = ((double)(rand()/randMaxDouble))*(maxi - mini) + mini;
    // printf("%lf \n", randomNumber);
    return randomNumber;
}

double sigmoid(double x){
    return (1.0/(1 + exp(-x)));
}

double relu(double x){
    return (x > 0) ? x : 0;
}

double tanh(double x){
    return ((exp(x) - exp(-x))/(exp(x) + exp(-x)));
}

double leakyRelu(double x, double leakage){
    NN_ASSERT(leakage >= 0);
    if(x >= 0) return x;
    return leakage*x;
}

double least_squared_error(double a, double b){
    return (a - b)*(a - b);
}

double cross_entropy_loss(double a, double b){
    return -(a*log(b) + (1-a)*log(1-b));
}

double cross_entropy_loss_derivative(double yi, double ai){
    return ((ai-yi)/(ai*(1-ai)));
}


Matrix matrix_alloc(int rows, int cols){
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.data = (double*)NN_MALLOC(sizeof(*(m.data))*rows*cols);
    NN_ASSERT(m.data != NULL);
    return m;
}

Matrix matrix_get_row(Matrix m, int row){
    //returns pointer to row, wrapped in a matrix
    NN_ASSERT(row < m.rows);
    return (Matrix){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .data = &MATRIX_AT(m, row, 0)
    }; 
}

void matrix_transpose(Matrix a, Matrix b){
    NN_ASSERT(a.rows == b.cols && b.rows == a.cols);
    for(int i = 0; i < b.rows; i++){
        for(int j = 0; j < b.cols; j++){
            MATRIX_AT(a, j, i) = MATRIX_AT(b, i, j);
        }
    }
}

Matrix matrix_sub_matrix(Matrix m, int startR, int endR, int startC, int endC){
    NN_ASSERT((m.rows > endR && m.cols > endC) && (startR >= 0 && startC >= 0));
    return (Matrix){
        .rows = endR - startR + 1,
        .cols = endC - startC + 1,
        .stride = m.stride,
        .data = &MATRIX_AT(m, startR, startC)
    };
}

void matrix_print(Matrix m, char* s){
    printf("%s [\n", s);
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            printf("%lf ", MATRIX_AT(m, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}

void matrix_print_nn(Matrix m, char* s, int padding){
    printf("%*s%s = [\n", (int) padding, "", s);
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            printf("%*s    %lf ",(int)padding, "", MATRIX_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int)padding, "");
}

void matrix_random_init(Matrix m, double mini, double maxi){
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            MATRIX_AT(m ,i, j) = generate_randoms(maxi, mini);
        }
    }
}


void matrix_multiply(Matrix destination, Matrix source1, Matrix source2){
    //in this case you must supply a destination matrix;
    //this is not optimal. not even fucking close.
    NN_ASSERT(source1.cols == source2.rows);
    NN_ASSERT((source1.rows == destination.rows) && (source2.cols == destination.cols));
    double d;
    for(int i = 0; i < source1.rows; i++){
        for(int j = 0; j < source2.cols; j++){
            d = 0;
            for(int k = 0; k < source1.cols; k++){
                d += MATRIX_AT(source1, i, k)*MATRIX_AT(source2, k, j); 
            }
            MATRIX_AT(destination, i, j) = d;
        }
    }
}

void matrix_multiply_element_wise(Matrix destination, Matrix a, Matrix b){
    NN_ASSERT((a.rows == b.rows && a.cols == b.cols));
    for(int i = 0; i < a.rows; i++){
        for(int j = 0; j < a.cols; j++){
            MATRIX_AT(destination, i, j) = MATRIX_AT(a, i, j) * MATRIX_AT(b, i, j);
        }
    }
}

void matrix_divide_element_wise(Matrix destination, Matrix a, Matrix b){
    NN_ASSERT((a.rows == b.rows && a.cols == b.cols));
    for(int i = 0; i < a.rows; i++){
        for(int j = 0; j < a.cols; j++){
            MATRIX_AT(destination, i, j) = MATRIX_AT(a, i, j) / MATRIX_AT(b, i, j);
        }
    }
}

void matrix_multiply_by_constant(Matrix m, double x){
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            MATRIX_AT(m, i, j) = MATRIX_AT(m, i, j)*x;
        }
    }
}

void matrix_add(Matrix destination, Matrix a, Matrix b){
    //optimized code for matrix addition
    NN_ASSERT((a.rows == b.rows && a.cols == b.cols));
    for(int i = 0; i < a.rows; i++){
        for(int j = 0; j < a.cols; j++){
            MATRIX_AT(destination, i, j) = MATRIX_AT(a, i, j) + MATRIX_AT(b, i, j);
        }
    }
}

void matrix_subtract(Matrix destination, Matrix a, Matrix b){
    NN_ASSERT((a.rows == b.rows && a.cols == b.cols));
    for(int i = 0; i < a.rows; i++){
        for(int j = 0; j < a.cols; j++){
            MATRIX_AT(destination, i, j) = MATRIX_AT(a, i, j) - MATRIX_AT(b, i, j);
        }
    }
}

void matrix_fill(Matrix a, double x){
    for(int i = 0; i < a.rows; i++){
        for(int j = 0; j < a.cols; j++){
            MATRIX_AT(a, i, j) = x;
        }
    }
}

void matrix_copy(Matrix a, Matrix b){
    NN_ASSERT((a.rows == b.rows) && (a.cols == b.cols));
    for(int i = 0; i < a.rows; i++){
        for(int j = 0; j < a.cols; j++){
            MATRIX_AT(a, i, j) = MATRIX_AT(b, i, j);
        }
    }
}

void matrix_sigmoid(Matrix a){
    for(int i = 0; i < a.rows; i++){
        for(int j = 0; j < a.cols; j++){
            MATRIX_AT(a, i, j) = sigmoid(MATRIX_AT(a, i, j));
        }
    }
}

void matrix_relu(Matrix a){
    for(int i = 0; i < a.rows; i++){
        for(int j = 0; j < a.cols; j++){
            MATRIX_AT(a, i, j) = relu(MATRIX_AT(a, i, j));
        }
    }
}

void matrix_leaky_relu(Matrix a, double leakage){
    for(int i = 0; i < a.rows; i++){
        for(int j = 0; j < a.cols; j++){
            MATRIX_AT(a, i, j) = leakyRelu(MATRIX_AT(a, i, j), leakage);
        }
    }
}


void matrix_tanh(Matrix a){
    for(int i = 0; i < a.rows; i++){
        for(int j = 0; j < a.cols; j++){
            MATRIX_AT(a, i, j) = tanh(MATRIX_AT(a, i, j));
        }
    }
}

void matrix_derivative_sigmoid(Matrix m){
    Matrix temp = matrix_alloc(m.rows, 1);
    matrix_fill(temp, 1);
    matrix_subtract(temp, temp, m);
    matrix_multiply_element_wise(m, temp, m);
}

void matrix_derivative_relu(Matrix m){
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            if(MATRIX_AT(m, i, j)) MATRIX_AT(m, i, j) = 1;                                  
        }
    }
}

void matrix_derivative_tanh(Matrix m){
    matrix_derivative_sigmoid(m);
    matrix_multiply_by_constant(m, 2);
}

void matrix_derivative_leaky_relu(Matrix m, double leakage){
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            MATRIX_AT(m, i, j) = (MATRIX_AT(m, i, j) >= 0)?1:leakage;
        }
    }
}

Layer layer_alloc(int curr, int prev, char* active){
    Layer l;
    if(prev == -1){
        l.activations = matrix_alloc(curr, 1);
    }
    else{
        l.weights = matrix_alloc(curr, prev);
        l.biases = matrix_alloc(curr, 1);
        l.activations = matrix_alloc(curr, 1);
    }
    l.activation = active;
    return l;

}   

void layer_copy(Layer p, Layer q){
    matrix_copy(p.weights, q.weights);
    matrix_copy(p.biases, q.biases);
    p.activation = q.activation;
}

void layer_activate(Layer l){
    char* relu = "relu";
    char* sigmoid = "sigmoid";
    char* tanh = "tanh";
    char* leakyRelu = "leakyRelu";
    if(strcmp(relu, l.activation) == 0){
        matrix_relu((l.activations));
    }
    if(strcmp(sigmoid, l.activation) == 0){
        matrix_sigmoid((l.activations));
    }
    if(strcmp(tanh, l.activation) == 0){
        matrix_tanh((l.activations));
    }
    if(strcmp(leakyRelu, l.activation) == 0){
        matrix_leaky_relu((l.activations), l.leakage);
    }
    return;
}

void layer_derivate(Layer l){
    char* relu = "relu";
    char* sigmoid = "sigmoid";
    char* tanh = "tanh";
    char* leakyRelu = "leakyRelu";
    if(strcmp(relu, l.activation) == 0){
        matrix_derivative_relu((l.activations));
    }
    if(strcmp(sigmoid, l.activation) == 0){
        matrix_derivative_sigmoid((l.activations));
    }
    if(strcmp(tanh, l.activation) == 0){
        matrix_derivative_tanh((l.activations));
    }
    if(strcmp(leakyRelu, l.activation) == 0){
        matrix_derivative_leaky_relu((l.activations), l.leakage);
    }
}



NeuralNetwork nn_alloc(int *arch, int layerCount, char** act){
    NN_ASSERT(layerCount != 0);
    srand(time(NULL));
    NeuralNetwork nn;
    nn.epsilon = 0.1;
    nn.alpha = 0.1;
    nn.activationFunctions = act;
    nn.layerCount = layerCount;
    char* broken = "linear";
 
    nn.architecture = (int*)NN_MALLOC(sizeof(int)*layerCount);
    nn.layers = (Layer*)NN_MALLOC(sizeof(Layer)*(layerCount));
    nn.architecture[0] = arch[0];
    nn.layers[0] = layer_alloc(arch[0], -1, broken);
    for(int i = 1; i < layerCount; i++){
        nn.layers[i] = layer_alloc(arch[i], arch[i-1], act[i]);
        nn.architecture[i] = arch[i];
    }
    return nn;
}

void nn_print(NeuralNetwork nn, char* s){
    char buffer[1024];
    printf("%s [\n", s);
    int layerCount = nn.layerCount-1;
    for(int i = 0; i < layerCount; i++){
        snprintf(buffer, sizeof(buffer), "weights%zu", i);
        matrix_print_nn(((nn.layers[i]).weights), buffer, 4);
    }
    printf("<----------------->\n");
    for(int i = 0; i < layerCount; i++){
        snprintf(buffer, sizeof(buffer), "biases%zu", i);
        matrix_print_nn(((nn.layers[i]).biases), buffer, 4);
    }
    printf("<----------------->\n");
    for(int i = 0; i < layerCount+1; i++){
        snprintf(buffer, sizeof(buffer), "activations%zu", i);
        matrix_print_nn(((nn.layers[i]).activations), buffer, 4);
    }
}

void nn_free(NeuralNetwork nn){
    int l = nn.layerCount;
    free((nn.layers[0]).activations.data);
    for(int i = 1; i < l; i++){
        free(((nn.layers[i]).weights).data);
        free(((nn.layers[i]).biases).data);
        free(((nn.layers[i]).activations).data);
    }
    free(nn.layers);
    free(nn.architecture); 
    return;
}

void nn_copy(NeuralNetwork a, NeuralNetwork b){
    NN_ASSERT(a.layerCount == b.layerCount);
    for(int i = 0; i < a.layerCount; i++){
        NN_ASSERT(a.architecture[i] == b.architecture[i]);
    }
    for(int i = 1; i < a.layerCount; i++){
        layer_copy((a.layers[i]), (b.layers[i]));
    }
}

void nn_random_init_all_weights(NeuralNetwork nn, double mini, double maxi){
    int layerCount = nn.layerCount; 
    for(int i = 1; i < layerCount; i++){
        matrix_random_init((nn.layers[i]).weights, mini, maxi);
        matrix_random_init((nn.layers[i]).biases, mini, maxi);
    }
}

void forward_pass(NeuralNetwork nn, Matrix transposedInput){
    NN_ASSERT(nn.layers[1].weights.cols == transposedInput.rows);
    matrix_copy((nn.layers[0].activations), transposedInput);
    int layerCount = nn.layerCount;
    for(int i = 1; i < layerCount; i++){
        matrix_multiply((nn.layers[i].activations), (nn.layers[i].weights), (nn.layers[i-1].activations));
        matrix_add((nn.layers[i].activations), (nn.layers[i].activations), (nn.layers[i].biases));
        layer_activate(nn.layers[i]);
    }
}


double nn_compute_cost(NeuralNetwork nn, Matrix input, Matrix output, int costFunction){
    NN_ASSERT(costFunction >= 0 && costFunction < 2);
    NN_ASSERT(input.rows == output.rows);
    NN_ASSERT((nn.layers[nn.layerCount-1].activations).rows == output.cols);
    NN_ASSERT(nn.layers[0].activations.rows == input.cols);
    int examples = input.rows;
    double cost = 0;
    double a, b, c;
    Matrix temp = matrix_alloc(input.cols, 1);
    for(int i = 0; i < examples; i++){
        matrix_transpose(temp, matrix_get_row(input, i));
        forward_pass(nn, temp);
        for(int j = 0; j < output.cols; j++){
            if(costFunction == 0) cost += least_squared_error(MATRIX_AT((nn.layers[nn.layerCount-1].activations), j, 0), MATRIX_AT(output, i, j));
            else cost += cross_entropy_loss(MATRIX_AT(output, i, j), MATRIX_AT((nn.layers[nn.layerCount-1].activations), j, 0));
        }
    }
    cost /= ((double)examples);
    free(temp.data);
    return cost;
}


void backpropagation(NeuralNetwork *nn, Matrix *input, Matrix *output, int iterations, int costFunction){
    NN_ASSERT((*input).rows == (*output).rows);
    NeuralNetwork gg = nn_alloc((*nn).architecture, (*nn).layerCount, (*nn).activationFunctions);
    int examples = (*input).rows;
    Matrix temp = matrix_alloc((*input).cols, 1);
    double cost;
    for(int k = 0; k < iterations; k++){
        
        for(int j = 0; j < examples; j++){
            cost = nn_compute_cost(*nn, *input, *output, costFunction);
            free(temp.data);    
            for(int j = 0; j < examples; j++){

            }
            if(costFunction == 0){
                matrix_multiply_by_constant((*nn).layers[(*nn).layerCount-1].activations, -1);
                matrix_add(gg.layers[gg.layerCount-1].activations, *output, (*nn).layers[(*nn).layerCount-1].activations);
                matrix_multiply_by_constant(gg.layers[gg.layerCount-1].activations, 2);
            } 
            else{
                Matrix costlyOperation = matrix_alloc((*output).cols, 1);
                matrix_transpose(costlyOperation, matrix_get_row((*output), j));
                matrix_subtract(gg.layers[gg.layerCount-1].activations, (*nn).layers[(*nn).layerCount-1].activations, costlyOperation);
                free(costlyOperation.data);

                temp = matrix_alloc((*output).cols, 1);
                matrix_fill(temp, 1);
                matrix_subtract(temp, temp, (*nn).layers[(*nn).layerCount - 1].activations);
                matrix_multiply_element_wise((*nn).layers[(*nn).layerCount - 1].activations, temp, (*nn).layers[(*nn).layerCount - 1].activations);
                matrix_divide_element_wise(gg.layers[gg.layerCount - 1].activations, gg.layers[gg.layerCount - 1].activations, (*nn).layers[(*nn).layerCount - 1].activations);
                free(temp.data);
            }
            for(int i = gg.layerCount-1; i > 0; i--){
                // Matrix t2 = matrix_alloc(gg.layers[i].activations.rows, 1);
                // matrix_copy(t2, (*nn).layers[i].activations);
                // matrix_derivative_sigmoid(t2);
                // matrix_multiply_element_wise(gg.layers[i].activations, gg.layers[i].activations, t2);
                // free(t2.data);
                layer_derivate((*nn).layers[i]);
                matrix_multiply_element_wise(gg.layers[i].activations, gg.layers[i].activations, (*nn).layers[i].activations);
                Matrix weightsT = matrix_alloc(gg.layers[i].weights.cols, gg.layers[i].weights.rows);
                for(int a = 0; a < gg.layers[i].weights.rows; a++){
                    for(int b = 0; b < gg.layers[i].weights.cols; b++){

                        ///oh moi error prone code that I shall henceforth work with on a mac
                        MATRIX_AT(gg.layers[i].weights, a, b) = MATRIX_AT((*nn).layers[i-1].activations, b, 0)*MATRIX_AT(gg.layers[i].activations, a, 0);
                    }
                }
                matrix_transpose(weightsT, (*nn).layers[i].weights);
                matrix_multiply(gg.layers[i-1].activations, weightsT, (*nn).layers[i].activations);
                free(weightsT.data);
            }
            for(int i = 1; i < gg.layerCount; i++){
                matrix_multiply_by_constant(gg.layers[i].weights, (*nn).alpha);
                matrix_subtract((*nn).layers[i].weights, (*nn).layers[i].weights, gg.layers[i].weights);
            }
        }
    }
}




void update_by_finite_difference(NeuralNetwork nn, Matrix input, Matrix output, int iterations, int costFunction){
    //I see. This is painfully slow, since you must calculate the cost variations by varying the weights
    NeuralNetwork gg = nn_alloc((nn.architecture), (nn.layerCount), (nn.activationFunctions));
    nn_copy(gg, nn);
    int layerCount = nn.layerCount;
    int rows, cols;
    double costPlusDeltaCost, store, delta, cost;
    for(int c = 0; c < iterations; c++){
        cost = nn_compute_cost(nn, input, output, costFunction);
        if(c == iterations-1){
            printf("Cost evaluated to %lf\n", cost);
        }
        for(int i = 1; i < layerCount; i++){
            rows = (nn.layers[i].weights).rows;
            cols = (nn.layers[i].weights).cols;
            for(int j = 0; j < rows; j++){
                for(int k = 0; k < cols; k++){
                    store = MATRIX_AT(((nn.layers[i].weights)), j, k);
                    MATRIX_AT(((nn.layers[i].weights)), j, k) = MATRIX_AT(((nn.layers[i].weights)), j, k) + nn.epsilon;
                    costPlusDeltaCost = nn_compute_cost(nn, input, output, costFunction);
                    MATRIX_AT(((nn.layers[i].weights)), j, k) = store;
                    delta = costPlusDeltaCost - cost;
                    delta = delta/(nn.epsilon);
                    MATRIX_AT(((gg.layers[i].weights)), j, k) = MATRIX_AT(((gg.layers[i].weights)), j, k) -  (nn.alpha)*delta;
                }
            }
            rows = nn.layers[i].biases.rows;
            for(int j = 0; j < rows; j++){
                store = MATRIX_AT(((nn.layers[i].biases)), j, 0);
                MATRIX_AT(((nn.layers[i].biases)), j, 0) = MATRIX_AT(((nn.layers[i].biases)), j, 0) + nn.epsilon;
                costPlusDeltaCost = nn_compute_cost(nn, input, output, costFunction);
                MATRIX_AT(((nn.layers[i].biases)), j, 0) = store; 
                delta = (costPlusDeltaCost - cost);
                delta = delta/(nn.epsilon);
                MATRIX_AT(((gg.layers[i].biases)), j, 0) = MATRIX_AT(((gg.layers[i].biases)), j, 0) - ((nn.alpha)*delta);
            }
        }
        nn_copy(nn, gg);
    }
    nn_free(gg);
}

void train(NeuralNetwork nn, Matrix input, Matrix output, int iterations, int costFunction){
    //you can choose the optimization technique   
    // update_by_finite_difference(nn, input, output, iterations, costFunction);
    backpropagation(&nn, &input, &output, iterations, costFunction);
    
}


#endif  
