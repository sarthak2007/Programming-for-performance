void testcase() {
    int cachePower = 20;
    int blockPower = 6;

    double[][][] X = new double[64][64][64];
    double[][][] A = new double[64][64][64];
    String cacheType = "DirectMapped";

    for(int k=0;k<64;k+=1){
        for(int j=0;j<64;j+=1){
            for(int i=0;i<64;i+=1){
                A[i][j][k] = 1 + X[k][j][i];
            }
        }
    }
}
// A -> 262144
// X -> 32768

void testcase() {
    int cachePower = 14;
    int blockPower = 5;
    int stride = 1;
    int N = 64;
    int[][][] A = new int[N][N][N];
    int[][][] B = new int[N][N][N];
    int[][][] C = new int[N][N][N];

    String cacheType = "FullyAssociative";

    for (int j = 0;j < N;j+=1){
        for(int i=0;i<N;i+=stride){
            for(int k=0;k<N;k+=1){
                C[i][k][j] += A[i][j][k] * B[k][j][i];
            }
        }
    }
}
// A -> 32768
// B -> 32768
// C -> 262144

void testcase() {
    int cachePower = 20;
    int blockPower = 6;

    double[][][] X = new double[64][64][64];
    double[][][] A = new double[64][64][64];
    String cacheType = "SetAssociative";
    int setSize = 8;

    for(int k=0;k<64;k+=1){
        for(int j=0;j<64;j+=1){
            for(int i=0;i<64;i+=1){
                A[i][j][k] = 1 + X[k][j][i];
            }
        }
    }
}
// A -> 262144
// X -> 32768

void testcase() {
    int cachePower = 15;
    int blockPower = 5;
    int stride = 1;
    int N = 64;
    int[][][] A = new int[N][N][N];
    int[][][] B = new int[N][N][N];
    int[][][] C = new int[N][N][N];

    String cacheType = "SetAssociative";
    int setSize = 8;

    for (int j = 0;j < N;j+=1){
        for(int i=0;i<N;i+=stride){
            for(int k=0;k<N;k+=1){
                C[i][k][j] += A[i][j][k] * B[k][j][i];
            }
        }
    }
}
// A -> 32768
// B -> 262144
// C -> 262144

void testcase11() {
    int cachePower = 14;
    int blockPower = 5;
    String cacheType = "SetAssociative";
    int setSize = 8;
    int[][][] A = new int[64][64][64];
    int sum = 0;
    for(int j = 0; j < 64; j += 4){
        for(int k = 0; k < 64; k += 2) {
            for(int i = 0; i < 64; i += 8) {
                sum += A[k][i][j];
            }
        }
    }
}
// A -> 4096

void testcase() {
    int cachePower = 20;
    int blockPower = 6;

    double[][][] X = new double[64][64][64];
    double[][][] A = new double[64][64][64];
    String cacheType = "DirectMapped";

    for(int k=0;k<64;k+=2){
        for(int j=0;j<64;j+=4){
            for(int i=0;i<64;i+=1){
                A[i][j][k] = 1 + X[k][j][i];
            }
        }
    }
}
// A -> 32768
// X -> 4096

void testcase() {
    int cachePower = 14;
    int blockPower = 5;
    int stride = 1;
    int N = 64;
    int[][][] A = new int[N][N][N];
    int[][][] B = new int[N][N][N];
    int[][][] C = new int[N][N][N];

    String cacheType = "FullyAssociative";

    for (int j = 0;j < N;j+=1){
        for(int i=0;i<N;i+=2){
            for(int k=0;k<N;k+=8){
                C[i][k][j] += A[i][j][k] * B[k][j][i];
            }
        }
    }
}
// A -> 16384
// B -> 4096
// C -> 2048

void testcase() {
    int cachePower = 20;
    int blockPower = 6;

    double[][][] X = new double[64][64][64];
    double[][][] A = new double[64][64][64];
    String cacheType = "SetAssociative";
    int setSize = 8;

    for(int k=0;k<64;k+=2){
        for(int j=0;j<64;j+=4){
            for(int i=0;i<64;i+=8){
                A[i][j][k] = 1 + X[k][j][i];
            }
        }
    }
}
// A -> 1024
// X -> 4096

void testcase() {
    int cachePower = 15;
    int blockPower = 5;
    int stride = 1;
    int N = 64;
    int[][][] A = new int[N][N][N];
    int[][][] B = new int[N][N][N];
    int[][][] C = new int[N][N][N];

    String cacheType = "SetAssociative";
    int setSize = 8;

    for (int j = 0;j < N;j+=2){
        for(int i=0;i<N;i+=2){
            for(int k=0;k<N;k+=2){
                C[i][k][j] += A[i][j][k] * B[k][j][i];
            }
        }
    }
}
// A -> 8192
// B -> 32768
// C -> 32768

void testcase() {
    int cachePower = 15;
    int blockPower = 5;
    int stride = 1;
    int N = 64;
    int[][][] A = new int[N][N][8];
    int[][][] B = new int[N][N][8];
    int[][][] C = new int[N][N][8];

    String cacheType = "SetAssociative";
    int setSize = 8;

    for (int j = 0;j < N;j+=2){
        for(int i=0;i<N;i+=2){
            for(int k=0;k<8;k+=2){
                C[i][j][k] += A[j][i][k] * B[i][j][k];
            }
        }
    }
}
// A -> 1024
// B -> 1024
// C -> 1024
