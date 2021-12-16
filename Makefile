OUT_FOLDER := bin

$(OUT_FOLDER)/final.out:  convolutionMulti.cu utils.cu layers/*.cu layers/*.h
	nvcc convolutionMulti.cu -Xptxas="-v" --use_fast_math -o $@

all: $(OUT_FOLDER)/final.out

clean:
	rm $(OUT_FOLDER)/*

.PHONY: test
test: $(OUT_FOLDER)/final.out
	$(OUT_FOLDER)/final.out
