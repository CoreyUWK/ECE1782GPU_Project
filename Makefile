OUT_FOLDER := bin
MY_NVCC_FLAGS = -Xptxas="-v" --use_fast_math

vpath %.cu testbench src src/layers
vpath %.h testbench src src/layers

$(OUT_FOLDER)/convolutionMulti.out: convolutionMulti.cu
	nvcc $^ $(MY_NVCC_FLAGS) -o $@

$(OUT_FOLDER)/simple.out: simple.cu
	nvcc $^ $(MY_NVCC_FLAGS) -o $@

$(OUT_FOLDER)/cnnParallel.out: cnnParallel.cu
	nvcc $^ $(MY_NVCC_FLAGS) -o $@

$(OUT_FOLDER)/final.out: convolutionMulti.cu
	nvcc $^ $(MY_NVCC_FLAGS) -o $@

$(OUT_FOLDER)/max_pool_2d.out: test_max_pool_2d.cu
	nvcc $^ $(MY_NVCC_FLAGS) -o $@

$(OUT_FOLDER)/relu.out: test_relu.cu
	nvcc $^ $(MY_NVCC_FLAGS) -o $@

$(OUT_FOLDER)/linear.out: test_linear.cu
	nvcc $^ $(MY_NVCC_FLAGS) -o $@

all: $(OUT_FOLDER)/convolutionMulti.out $(OUT_FOLDER)/simple.out $(OUT_FOLDER)/cnnParallel.out $(OUT_FOLDER)/max_pool_2d.out $(OUT_FOLDER)/relu.out $(OUT_FOLDER)/linear.out

clean:
	rm $(OUT_FOLDER)/*

.PHONY: test
test: all
	$(OUT_FOLDER)/convolutionMulti.out
	$(OUT_FOLDER)/simple.out
	$(OUT_FOLDER)/cnnParallel.out
	$(OUT_FOLDER)/max_pool_2d.out 10 10
	$(OUT_FOLDER)/relu.out 10 10
	$(OUT_FOLDER)/linear.out
