OUT_FOLDER := bin

vpath %.cu testbench src src/layers
vpath %.h testbench src src/layers

$(OUT_FOLDER)/final.out: convolutionMulti.cu
	nvcc $^ -Xptxas="-v" --use_fast_math -o $@

$(OUT_FOLDER)/max_pool_2d.out: test_max_pool_2d.cu
	nvcc $^ -Xptxas="-v" --use_fast_math -o $@

all: $(OUT_FOLDER)/final.out $(OUT_FOLDER)/max_pool_2d.out

clean:
	rm $(OUT_FOLDER)/*

.PHONY: test
test: $(OUT_FOLDER)/final.out
	$(OUT_FOLDER)/final.out
	$(OUT_FOLDER)/max_pool_2d.out 10 10
