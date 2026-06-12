NVCC = nvcc
CXXFLAGS = -O3 -std=c++17 -arch=sm_80 --use_fast_math

TARGET = einstein_validator_v1
SRC = einstein_validator_v1.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET)

triage: $(TARGET)
	./$(TARGET) \
	  --input records_einstein_v3_very_hard \
	  --out validation_triage_v1 \
	  --period 14 \
	  --nodes 1000000 \
	  --patch-radius 72 \
	  --random-trials 120 \
	  --boundary-trials 80 \
	  --forced-trials 80 \
	  --region-min 3 \
	  --region-max 5 \
	  --region-nodes 500000 \
	  --weak-tiles 120 \
	  --strong-tiles 280 \
	  --very-strong-tiles 500

hard: $(TARGET)
	./$(TARGET) \
	  --input records_einstein_v3_very_hard \
	  --out validation_hard_v1 \
	  --period 18 \
	  --nodes 20000000 \
	  --patch-radius 120 \
	  --random-trials 1000 \
	  --boundary-trials 700 \
	  --forced-trials 700 \
	  --region-min 3 \
	  --region-max 8 \
	  --region-nodes 5000000 \
	  --weak-tiles 180 \
	  --strong-tiles 500 \
	  --very-strong-tiles 900

clean:
	rm -f $(TARGET)