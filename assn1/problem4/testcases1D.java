void testcase() {
    int cachePower = 16;
    int blockPower = 5;
    int stride = 1;
    int N = 1024;
    long[] A = new long[N];
    String cacheType = "DirectMapped";

    for(int j = 0; j<1024; j+=8) {
    	for (int i = 0;i < N;i+=1){
        	A[i] = 0;
    	}
    }
}
// 256

void testcase() {
    int cachePower = 16;
    int blockPower = 5;
    int stride = 1;
    int N = 1024;
    long[] A = new long[N];
    String cacheType = "DirectMapped";

    for(int j = 0; j<1024; j+=8) {
    	for (int i = 0;i < N;i+=N){
        	A[i] = 0;
    	}
    }
}
// 1

void testcase() {
    int cachePower = 16;
    int blockPower = 5;
    int stride = 1;
    int N = 1024;
    long[] A = new long[N];
    String cacheType = "DirectMapped";

    for(int j = 0; j<1024; j+=8) {
    	for (int i = 0;i < N;i+=8){
        	A[i] = 0;
    	}
    }
}
// 128

void testcase() {
    int cachePower = 11;
    int blockPower = 5;
    int stride = 1;
    int N = 1024;
    long[] A = new long[N];
    String cacheType = "FullyAssociative";

    for(int j = 0; j<1024; j+=8) {
    	for (int i = 0;i < N;i+=16){
        	A[i] = 0;
    	}
    }
}
// 64

void testcase() {
    int cachePower = 11;
    int blockPower = 5;
    int stride = 1;
    int N = 1024;
    long[] A = new long[N];
    String cacheType = "FullyAssociative";

    for(int j = 0; j<1024; j+=8) {
    	for (int i = 0;i < N;i+=8){
        	A[i] = 0;
    	}
    }
}
// 16384

void testcase() {
    int cachePower = 11;
    int blockPower = 5;
    int stride = 1;
    int N = 1024;
    long[] A = new long[N];
    String cacheType = "SetAssociative";
    long setSize = 4;

    for(int j = 0; j<1024; j+=8) {
    	for (int i = 0;i < N;i+=128){
        	A[i] = 0;
    	}
    }
}
// 1024

void testcase() {
    int cachePower = 11;
    int blockPower = 5;
    int stride = 1;
    int N = 1024;
    long[] A = new long[N];
    String cacheType = "SetAssociative";
    long setSize = 4;

    for(int j = 0; j<1024; j+=8) {
    	for (int i = 0;i < N;i+=256){
        	A[i] = 0;
    	}
    }
}
// 4

void testcase() {
    int cachePower = 11;
    int blockPower = 5;
    int stride = 1;
    int N = 1024;
    long[] A = new long[N];
    String cacheType = "SetAssociative";
    long setSize = 4;

    for(int k = 0; k<2; k+=1) {
        for(int j = 0; j<1024; j+=8) {
            for (int i = 0;i < N;i+=1){
                A[i] = 0;
            }
        }
    }
}
// 65536

void testcase() {
    int cachePower = 18;
    int blockPower = 5;
    int stride = 1;
    int ten = 320;
    int N = 32768;
    double[] A = new double[N];
    String cacheType = "SetAssociative";
    int setSize = 4;
    double s = 0.0;
    for (int i = 0;i < ten;i+=1){
        for(int j=0;j<N;j+=stride){
            s += A[j];
        }
    }
}
// 8192