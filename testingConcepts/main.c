#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// typedef double sample[3];


//reinventing the wheel
//this model uses finite difference to calculate derivatives instead of calculating the derivatives using the actual formulas

//The following is a realization of an XOR gate by training the models for OR and AND gates

    double sigmoid(double x){
        return 1/(1 + exp(-x));
    }





    // void initDataSet(double (*arr)[2], int n){
    //     for(int i = 0; i < n; i++){
    //         arr[i][0] = i;
    //         arr[i][1] = 2*i;
    //     }
    // }

    double generate_param(int seeb){
        srand(seeb);
        double x = (double)(rand()%10);
        return x;
    }


    double cost(double w1, double w2, double b, double (*dataset)[3]){
        double running_sum = 0;
        int x = 4;
        for(int i = 0; i < x; i++){
            double temp = (sigmoid((w1*dataset[i][0] + w2*dataset[i][1] + b)) - dataset[i][2]);
            temp *= temp;
            running_sum += temp;
        } 
        running_sum /= (double)x;
        return running_sum;
    }


    void updateWeights(double *w1, double *w2, double *b, double (*dataset)[3]){
        const double eps = 0.01;
        const double learning_rate = 0.01;
        double t1 = *w1;
        double t2 = *w2;
        double t3 = *b;
        double c1 = cost(t1+eps, t2, t3, dataset);
        double c2 = cost(t1, t2 + eps, t3, dataset);
        double c3 = cost(t1, t2, t3 + eps, dataset);
        double c = cost(t1, t2, t3, dataset);
        // printf("%lf %lf\n", c1, c2);
        double delta_cost1 = (c1 - c)/eps;
        double delta_cost2 = (c2 - c)/eps;
        double delta_cost3 = (c3 - c)/eps;
        *w1 = *w1 - learning_rate*delta_cost1;
        *w2 = *w2 - learning_rate*delta_cost2;
        *b = *b - learning_rate*delta_cost3;
        // printf("Updated parameters to %lf %lf %lf\n", *w1, *w2, *b);
        return;
    }


    int main(){
        double dataset[4][3];
        double dataset2[4][3];
        // initDataSet(dataset, 10);
        int count  = 0;
        for(int i = 0; i < 2; i++){
            for(int j = 0; j < 2; j++){
                dataset[i+j+count][0] = i;
                dataset[i+j+count][1] = j;
                dataset[i+j+count][2] = i&j;

            }
            count++;
        }
        count = 0;
        for(int i = 0; i < 2; i++){
            for(int j = 0; j < 2; j++){
                dataset2[i+j+count][0] = i;
                dataset2[i+j+count][1] = j;
                dataset2[i+j+count][2] = i|j;

            }
            count++;
        }
        // for(int i = 0; i < 4; i++){
        //     for(int j = 0; j < 3; j++){
        //         printf("%lf ", dataset[i][j]);
        //     }
        //     printf("\n");
        // }
        printf("Hello Ishaan\n");
        int train_count = 10000000;
        double w1 = generate_param(1);
        double w2 = generate_param(10);
        double b = generate_param(69);
        double w3 = generate_param(2);
        double w4 = generate_param(13);
        double b2 = generate_param(420);
        for(int i = 0; i < train_count; i++){
            updateWeights(&w1, &w2, &b, dataset);
        }
        for(int i = 0; i < train_count; i++){
            updateWeights(&w3, &w4, &b2, dataset2);
        }
        double a , f;
        printf("%lf %lf %lf\n", w1, w2, b);
        printf("%lf %lf %lf\n", w3, w4, b2);
        scanf("%lf %lf", &a, &f);
        double ans = sigmoid(w3*a + w4*f + b2) * (1 - sigmoid(w1*a + w2*f + b));
        printf("%lf", ans);
        return 0;
    }



// artificial neurons modelled



