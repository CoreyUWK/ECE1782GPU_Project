float *softmax(int size, float *z) {
    float max = 0;
    for (int i = 0; i < size; i++) {
        if (z[i] > max) {
            max = z[i];
        }
    }
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += expf(z[i] - max);
    }
    float *buff = new float[size];
    for (int i = 0; i < size; i++) {
        buff[i] = expf(z[i] - max) / sum;
    }
    return buff;
}
