# Don't use the --user flag for setup.py develop mode with virtualenv.
.PHONY: default
default: dev

.PHONY: install
install:
	python setup.py install
.PHONY: ops
ops:
#	mkdir -p build && cd build && CC=gcc CXX=g++ cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0-cudnn-6.0 .. && CC=gcc CXX=g++ make -j$(shell nproc)
	mkdir -p build && cd build && CC=gcc-5 CXX=g++-5 cmake .. -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF -DCMAKE_BUILD_TYPE=Release && CC=gcc-5 CXX=g++-5 make -j$(shell nproc)
.PHONY: dev
dev:
	CC=gcc CXX=g++ python setup.py develop
.PHONY: clean
clean:
	python setup.py develop --uninstall
	rm -rf build
