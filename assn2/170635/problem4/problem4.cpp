/**
 * g++ -o problem4 problem4.cpp -pthread
 * ./problem4 <number of customers>
 */

#include<iostream>
#include<pthread.h>
#include<queue>
#include<unistd.h>

const int NUM_THREADS = 24;
const int NUM_TELLERS = 2;

pthread_mutex_t token_lock;
pthread_mutex_t queue_lock;
pthread_cond_t queue_cv;
pthread_cond_t shared_array_cv;
pthread_mutex_t shared_array_lock;
pthread_mutex_t array_elements_lock[16];
pthread_mutex_t print_lock;
pthread_cond_t customer_to_teller_cv;
pthread_mutex_t customer_to_teller_lock;
pthread_cond_t teller_to_customer_cv;
pthread_mutex_t teller_to_customer_lock;
std::queue<int> q;
int shared_array[16];
int global_token = 1;
int teller_tokens[NUM_TELLERS] = {0, 0};
int written[NUM_TELLERS] = {0, 0};
int N, cnt_customers;

void* customer(void* temp){
    int token;
    pthread_mutex_lock(&token_lock);
    token = global_token++;
    pthread_mutex_unlock(&token_lock);

    pthread_mutex_lock(&queue_lock);
    q.push(token);
    pthread_cond_broadcast(&queue_cv);
    pthread_mutex_unlock(&queue_lock);

    int teller;
    pthread_mutex_lock(&shared_array_lock);
    while(teller_tokens[0] != token and teller_tokens[1] != token){
        pthread_cond_wait(&shared_array_cv, &shared_array_lock);
    }
    if(teller_tokens[0] == token)
        teller = 0;
    else
        teller = 1;
    pthread_mutex_unlock(&shared_array_lock);
    
    int cnt = 0;
    for(int i=0;i<16;i++){
        if(cnt == 8)
            break;
        pthread_mutex_lock(&array_elements_lock[i]);
        if(shared_array[i] == 0){
            shared_array[i] = token;
            cnt++;
        }
        pthread_mutex_unlock(&array_elements_lock[i]);
    }

    pthread_mutex_lock(&customer_to_teller_lock);
    written[teller] = 1;
    pthread_cond_broadcast(&customer_to_teller_cv);
    pthread_mutex_unlock(&customer_to_teller_lock);

    pthread_mutex_lock(&teller_to_customer_lock);
    while(teller_tokens[teller] == token){
        pthread_cond_wait(&teller_to_customer_cv, &teller_to_customer_lock);
    }
    pthread_mutex_unlock(&teller_to_customer_lock);   

    pthread_exit(NULL);
}

void* teller(void* thread_id){
    int teller_id = (intptr_t)thread_id;

    while(cnt_customers != 0){
        pthread_mutex_lock(&shared_array_lock);
        if(cnt_customers == 0){
            pthread_mutex_unlock(&shared_array_lock);
            break;
        }
        pthread_mutex_lock(&queue_lock);    
        while(q.empty())
            pthread_cond_wait(&queue_cv, &queue_lock);
        teller_tokens[teller_id] = q.front();
        q.pop();
        cnt_customers--;
        pthread_mutex_unlock(&queue_lock);    
        pthread_cond_broadcast(&shared_array_cv);
        pthread_mutex_unlock(&shared_array_lock);

        pthread_mutex_lock(&customer_to_teller_lock);
        while(written[teller_id] == 0)
            pthread_cond_wait(&customer_to_teller_cv, &customer_to_teller_lock);
        pthread_mutex_unlock(&customer_to_teller_lock);
        
        pthread_mutex_lock(&print_lock);    
        for(int i=0;i<16;i++){
            pthread_mutex_lock(&array_elements_lock[i]);
            if(teller_tokens[teller_id] == shared_array[i]){
                std::cout << shared_array[i] << " ";
                shared_array[i] = 0;
            }
            pthread_mutex_unlock(&array_elements_lock[i]);
        }
        std::cout << std::endl; 
        pthread_mutex_unlock(&print_lock); 

        pthread_mutex_lock(&teller_to_customer_lock);
        teller_tokens[teller_id] = 0;
        written[teller_id] = 0;
        pthread_cond_broadcast(&teller_to_customer_cv);
        pthread_mutex_unlock(&teller_to_customer_lock);        

        sleep(5);

    }

    pthread_exit(NULL);
}


int main(int argc, char *argv[]){
    if(argc != 2){
        std::cout << "Usage: " << argv[0] << " <#customers>";
        exit(-1);
    }

    N = atoi(argv[1]);
    cnt_customers = N;
    if(N < 0 or N > NUM_THREADS){
        std::cout << "Number of customers must be between 1-" << NUM_THREADS;
        exit(-1);
    }

    pthread_t tellers[NUM_TELLERS];
    pthread_t customers[NUM_THREADS];
    pthread_attr_t attr;

    pthread_mutex_init(&token_lock, NULL);
    pthread_mutex_init(&queue_lock, NULL);
    pthread_cond_init(&queue_cv, NULL);
    pthread_cond_init(&shared_array_cv, NULL);
    pthread_mutex_init(&shared_array_lock, NULL);
    for(int i=0;i<16;i++)
        pthread_mutex_init(&array_elements_lock[i], NULL);
    pthread_mutex_init(&print_lock, NULL);
    pthread_cond_init(&customer_to_teller_cv, NULL);
    pthread_mutex_init(&customer_to_teller_lock, NULL);
    pthread_cond_init(&teller_to_customer_cv, NULL);
    pthread_mutex_init(&teller_to_customer_lock, NULL);

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    for(int i=0;i<16;i++)
        shared_array[i] = 0;

    for(int i=0;i<N;i++){
        if(pthread_create(&customers[i], &attr, customer, NULL)){
            std::cout << "Error in creating customers threads";
            exit(-1);
        }
    }
    for(int i=0;i<NUM_TELLERS;i++){
        if(pthread_create(&tellers[i], &attr, teller, (void*)(intptr_t)(i))){
            std::cout << "Error in creating teller threads";
            exit(-1);
        }
    }

    for(int i=0;i<N;i++){
        pthread_join(customers[i], NULL);
    }
    for(int i=0;i<NUM_TELLERS;i++){
        pthread_join(tellers[i], NULL);
    }
    
    pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&token_lock);
    pthread_mutex_destroy(&queue_lock);
    pthread_cond_destroy(&queue_cv);
    pthread_cond_destroy(&shared_array_cv);
    pthread_mutex_destroy(&shared_array_lock);
    for(int i=0;i<16;i++)
        pthread_mutex_destroy(&array_elements_lock[i]);
    pthread_mutex_destroy(&print_lock);
    pthread_cond_destroy(&customer_to_teller_cv);
    pthread_mutex_destroy(&customer_to_teller_lock);
    pthread_cond_destroy(&teller_to_customer_cv);
    pthread_mutex_destroy(&teller_to_customer_lock);
    pthread_exit(NULL);
    
    return 0;
}