void testcase() {
    int cachePower = 24;
    int blockPower = 6;

    double[] y = new double[4096];
    double[][] X = new double[4096][4096];
    double[][] A = new double[4096][4096];
    String cacheType = "DirectMapped";

    for(int k=0;k<4096;k+=1){
        for(int j=0;j<4096;j+=1){
            for(int i=0;i<4096;i+=1){
                y[i] += A[i][j] * X[k][j];
            }
        }
    }
}
// A -> 68719476736
// X -> 2097152
// y -> 512

void testcase() {
    int cachePower = 17;
    int blockPower = 5;
    int stride = 1;
    int N = 512;
    int[][] A = new int[N][N];
    int[][] B = new int[N][N];
    int[][] C = new int[N][N];

    String cacheType = "FullyAssociative";

    for (int j = 0;j < N;j+=1){
        for(int i=0;i<N;i+=stride){
            for(int k=0;k<N;k+=1){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
// A -> 16777216
// B -> 32768
// C -> 32768

void testcase() {
    int cachePower = 17;
    int blockPower = 5;
    int stride = 1;
    int N = 512;
    int[][] A = new int[N][N];
    int[][] B = new int[N][N];
    int[][] C = new int[N][N];

    String cacheType = "DirectMapped";

    for (int j = 0;j < N;j+=1){
        for(int i=0;i<N;i+=stride){
            for(int k=0;k<N;k+=1){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
// A -> 16777216
// B -> 134217728
// C -> 262144

void testcase() {
    int cachePower = 17;
    int blockPower = 5;
    int stride = 1;
    int N = 512;
    int[][] A = new int[N][N];
    int[][] B = new int[N][N];
    int[][] C = new int[N][N];

    String cacheType = "DirectMapped";

    for (int i = 0;i < N;i+=1){
        for(int k=0;k<N;k+=stride){
            for(int j=0;j<N;j+=1){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
// A -> 32768
// B -> 16777216
// C -> 32768

void testcase() {
    int cachePower = 17;
    int blockPower = 5;
    int stride = 1;
    int N = 512;
    int[][] A = new int[N][N];
    int[][] B = new int[N][N];
    int[][] C = new int[N][N];

    String cacheType = "DirectMapped";

    for (int i = 0;i < N;i+=1){
        for(int k=0;k<N;k+=stride){
            for(int j=0;j<N;j+=1){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
// A -> 32768
// B -> 16777216
// C -> 32768

void testcase() {
    int cachePower = 10;
    int blockPower = 5;

    long[][] A = new long[8][2];

    String cacheType = "DirectMapped";

    for(int i = 0; i<8; i+=1) {
        for(int j = 0; j<2; j+=1) {
            A[i][j] = 0;
        }
    }
}
// A -> 4

void testcase() {
    int cachePower = 10;
    int blockPower = 2;

    long[][] A = new long[8][2];

    String cacheType = "DirectMapped";

    for(int i = 0; i<8; i+=1) {
        for(int j = 0; j<2; j+=1) {
            A[i][j] = 0;
        }
    }
}
// A -> 16

void testcase() {
    int cachePower = 10;
    int blockPower = 5;

    long[][] A = new long[8][2];

    String cacheType = "DirectMapped";

    for(int i = 0; i<8; i+=2) {
        for(int j = 0; j<2; j+=1) {
            A[i][j] = 0;
        }
    }
}
// A -> 4
