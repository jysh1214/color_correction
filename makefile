.PHONY: clean dirs doc test

OPENMP=1
CUDA=1

SRC=src
OBJ=obj
TEST=test
OPENCV=`pkg-config --libs opencv4` `pkg-config --cflags opencv4`
CLANG=clang++ -Wall -Wunused-function -Wunused-variable -std=c++17 -lstdc++
NVCC=nvcc -DCUDA -lcuda -lcudart -lcublas -lcurand
OBJS=

ifeq ($(OPENMP), 1)
CLANG+=-Xpreprocessor -fopenmp -lomp
endif

ifeq ($(CUDA), 1)
CLANG+=-DCUDA -lcuda -lcudart -lcublas -lcurand
OBJS+=$(OBJ)/color_correction.o $(OBJ)/hist.o 
endif

all: dirs color_correction


color_correction: $(SRC)/main.cpp $(OBJS)
	$(CLANG) $^ -o $@ $(OPENCV)

$(OBJ)/color_correction.o: $(SRC)/color_correction.cu $(SRC)/color_correction.h
	$(NVCC) -c $< -o $@ $(OPENCV)

$(OBJ)/hist.o: $(SRC)/hist.cu $(SRC)/hist.h
	$(NVCC) -c $< -o $@ $(OPENCV)

clean:
	rm -rf $(OBJ)

dirs:
	mkdir -p $(SRC) $(OBJ)

stat:
	wc $(SRC)/*