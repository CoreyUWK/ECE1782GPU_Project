#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>

int load_weights(char * src, float *dst, unsigned int max_len) {
	FILE *fp;

	fp = fopen(src, "r");

	if (fp == NULL) {
		printf("Error when opening %s", src);
		exit(1);
	}

	for (int i = 0; i < max_len; i++) {
		if (fscanf(fp, "%f", &dst[i]) == EOF) {
			printf(
				"Error, file %s shorter than expected. Expected: %i Got: %i\n",
				src,
				max_len,
				i
			);
			exit(1);
		}
	}

	fclose(fp);

	return 0;
}
