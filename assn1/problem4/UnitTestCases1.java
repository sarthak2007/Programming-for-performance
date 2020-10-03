void testcase1() {
    int cachePower = 17; // cache size = 2^16B
    int blockPower = 5; // block size = 2^5B
    int stride = 1;
    int N = 512;
    int[][] A = new int[N][N];
    int[][] B = new int[N][N];
    int[][] C = new int[N][N];

    String cacheType = "DirectMapped";
    // int setSize = 4;
    // double s = 0.0;
    for (int i = 0;i < N;i+=1){
        for(int k=0;k<N;k+=stride){
            for(int j=0;j<N;j+=1){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void testcase1() {
    int cachePower = 17; // cache size = 2^16B
    int blockPower = 5; // block size = 2^5B
    int stride = 1;
    int N = 512;
    int[][] A = new int[N][N];
    int[][] B = new int[N][N];
    int[][] C = new int[N][N];

    String cacheType = "FullyAssociative";
    // int setSize = 4;
    // double s = 0.0;
    for (int i = 0;i < N;i+=1){
        for(int k=0;k<N;k+=stride){
            for(int j=0;j<N;j+=1){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
void testcase1() {
    int cachePower = 17; // cache size = 2^16B
    int blockPower = 5; // block size = 2^5B
    int stride = 1;
    int N = 512;
    int[][] A = new int[N][N];
    int[][] B = new int[N][N];
    int[][] C = new int[N][N];

    String cacheType = "DirectMapped";
    // int setSize = 4;
    // double s = 0.0;
    for (int j = 0;j < N;j+=1){
        for(int i=0;i<N;i+=stride){
            for(int k=0;k<N;k+=1){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void testcase1() {
    int cachePower = 17; // cache size = 2^16B
    int blockPower = 5; // block size = 2^5B
    int stride = 1;
    int N = 512;
    int[][] A = new int[N][N];
    int[][] B = new int[N][N];
    int[][] C = new int[N][N];

    String cacheType = "FullyAssociative";
    // int setSize = 4;
    // double s = 0.0;
    for (int j = 0;j < N;j+=1){
        for(int i=0;i<N;i+=stride){
            for(int k=0;k<N;k+=1){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void testcase1() {
    int cachePower = 24; // cache size = 2^16B
    int blockPower = 6; // block size = 2^5B
    int stride = 1;
    int ten = 10;
    int N = 32768;
    double[] y = new double[4096];
    double[][] X = new double[4096][4096];
    double[][] A = new double[4096][4096];
    String cacheType = "DirectMapped";
    int setSize = 4;
    double s = 0.0;
    for(int k=0;k<4096;k+=1){
        for(int j=0;j<4096;j+=1){
            for(int i=0;i<4096;i+=1){
                y[i] += A[i][j] * X[k][j];
            }
        }
    }
}

// void testcase1() {
//     int cachePower = 18; // cache size = 2^16B
//     int blockPower = 5; // block size = 2^5B
//     int stride = 1;
//     int ten = 10;
//     int N = 32768;
//     double[] A = new double[N];
//     String cacheType = "SetAssociative";
//     int setSize = 4;
//     double s = 0.0;
//     for (int i = 0;i < ten;i+=1){
//         for(int j=0;j<N;j+=stride){
//             s += A[j];
//         }
//     }
// }

// void testcase1() {
//     int cachePower = 18; // cache size = 2^16B
//     int blockPower = 5; // block size = 2^5B
//     int stride = 4;
//     int ten = 40;
//     int N = 32768;
//     double[] A = new double[N];
//     String cacheType = "SetAssociative";
//     int setSize = 4;
//     double s = 0.0;
//     for (int i = 0;i < ten;i+=1){
//         for(int j=0;j<N;j+=stride){
//             s += A[j];
//         }
//     }
// }

// void testcase1() {
//     int cachePower = 18; // cache size = 2^16B
//     int blockPower = 5; // block size = 2^5B
//     int stride = 16;
//     int ten = 160;
//     int N = 32768;
//     double[] A = new double[N];
//     String cacheType = "SetAssociative";
//     int setSize = 4;
//     double s = 0.0;
//     for (int i = 0;i < ten;i+=1){
//         for(int j=0;j<N;j+=stride){
//             s += A[j];
//         }
//     }
// }

// void testcase1() {
//     int cachePower = 18; // cache size = 2^16B
//     int blockPower = 5; // block size = 2^5B
//     int stride = 32;
//     int ten = 320;
//     int N = 32768;
//     double[] A = new double[N];
//     String cacheType = "SetAssociative";
//     int setSize = 4;
//     double s = 0.0;
//     for (int i = 0;i < ten;i+=1){
//         for(int j=0;j<N;j+=stride){
//             s += A[j];
//         }
//     }
// }void testcase1() {
//     int cachePower = 18; // cache size = 2^16B
//     int blockPower = 5; // block size = 2^5B
//     int stride = 2048;
//     int ten = 20480;
//     int N = 32768;
//     double[] A = new double[N];
//     String cacheType = "SetAssociative";
//     int setSize = 4;
//     double s = 0.0;
//     for (int i = 0;i < ten;i+=1){
//         for(int j=0;j<N;j+=stride){
//             s += A[j];
//         }
//     }
// }

// void testcase1() {
//     int cachePower = 18; // cache size = 2^16B
//     int blockPower = 5; // block size = 2^5B
//     int stride = 8192;
//     int ten = 81920;
//     int N = 32768;
//     double[] A = new double[N];
//     String cacheType = "SetAssociative";
//     int setSize = 4;
//     double s = 0.0;
//     for (int i = 0;i < ten;i+=1){
//         for(int j=0;j<N;j+=stride){
//             s += A[j];
//         }
//     }
// }
// void testcase1() {
//     int cachePower = 18; // cache size = 2^16B
//     int blockPower = 5; // block size = 2^5B
//     int stride = 32768;
//     int ten = 327680;
//     int N = 32768;
//     double[] A = new double[N];
//     String cacheType = "SetAssociative";
//     int setSize = 4;
//     double s = 0.0;
//     for (int i = 0;i < ten;i+=1){
//         for(int j=0;j<N;j+=stride){
//             s += A[j];
//         }
//     }
// }