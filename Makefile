final.out:  convolutionMulti.cu utils.cu layers/*.cu layers/*.h
	nvcc convolutionMulti.cu -Xptxas="-v" --use_fast_math -o final.out

all: final.out

clean:
	rm final.out

.PHONY: test
test: final.out
	./final.out
