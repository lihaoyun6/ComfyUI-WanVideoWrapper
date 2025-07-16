# ComfyUI-WanVideoWrapper
## This project only supports NVIDIA RTX30/40 and A100/H100 GPU!

This is a modified version of ComfyUI-WanVideoWrapper.  
 I
added **[draft-attention](https://github.com/shawnricecake/draft-attention)** support and **[Euler/d](https://github.com/eddyhhlure1Eddy/Euler-d)** scheduler to ComfyUI-WanVideoWrapper to further speed up video inference.  
And this update significantly improves the inference speed of NVIDIA Ampere GPU (sm80).  
The average inference speed is increased by more than 30% compared to sage-attention.  
**[[中文版本](./readme_zh.md)]** **[[original readme](./original_readme.md)]**  

## Installation

### Pre-Build Installation:    
1. Install flash-attention and Block-Sparse-Attention.   
	> There are only wheels for `win_amd64` here!  
	> It's recommended to update NVIDIA driver and install `PyTorch 2.7.x` and install wheels according to your python version

	```bash
	# Python 3.10
	pip install https://github.com/lihaoyun6/Block-Sparse-Attention/releases/download/v0.0.1/flash_attn-2.8.1+cu128torch2.7cxx11abiFALSE-cp310-cp310-win_amd64.whl
	pip install https://github.com/lihaoyun6/Block-Sparse-Attention/releases/download/v0.0.1/block_sparse_attn-0.0.1+cu128torch2.7cxx11abiFALSE-cp310-cp310-win_amd64.whl
	```
	
	```bash
	# Python 3.11
	pip install https://github.com/lihaoyun6/Block-Sparse-Attention/releases/download/v0.0.1/flash_attn-2.8.1+cu128torch2.7cxx11abiFALSE-cp311-cp311-win_amd64.whl
	pip install https://github.com/lihaoyun6/Block-Sparse-Attention/releases/download/v0.0.1/block_sparse_attn-0.0.1+cu128torch2.7cxx11abiFALSE-cp311-cp311-win_amd64.whl
	```

	```bash
	# Python 3.12
	pip install https://github.com/lihaoyun6/Block-Sparse-Attention/releases/download/v0.0.1/flash_attn-2.8.1+cu128torch2.7cxx11abiFALSE-cp312-cp312-win_amd64.whl
	pip install https://github.com/lihaoyun6/Block-Sparse-Attention/releases/download/v0.0.1/block_sparse_attn-0.0.1+cu128torch2.7cxx11abiFALSE-cp312-cp312-win_amd64.whl
	```

	```bash
	# Python 3.13
	pip install https://github.com/lihaoyun6/Block-Sparse-Attention/releases/download/v0.0.1/flash_attn-2.8.1+cu128torch2.7cxx11abiFALSE-cp313-cp313-win_amd64.whl
	pip install https://github.com/lihaoyun6/Block-Sparse-Attention/releases/download/v0.0.1/block_sparse_attn-0.0.1+cu128torch2.7cxx11abiFALSE-cp313-cp313-win_amd64.whl
	```

2. Clone this repo and install the dependencies:  

	```bash
	cd [ComfyUI_Path]\custom_nodes
	git clone https://github.com/lihaoyun6/ComfyUI-WanVideoWrapper
	pip install -r ComfyUI-WanVideoWrapper\requirements.txt
	```

---

### Manual Installation:
1. Build and install Block-Sparse-Attention:
	> Please make sure you have installed the CUDA Toolkit, C++ compiler and PyTorch  

	```bash
	pip install ninja wheel packaging
	#Each JOB will use about 8GB of RAM, please set it appropriately.
	#For example, if the you have 64GB RAM, you can set MAX_JOBS=8
	export MAX_JOBS=4
	#Download source code and install.
	#This process may take 1~3 hours depending on your CPU and RAM. Please be patient.
	git clone https://github.com/lihaoyun6/Block-Sparse-Attention
	cd Block-Sparse-Attention
	pip install --no-build-isolation ./
	```

2. Install flash-attention:  

	```bash
	pip install flash-attention --no-build-isolation
	```

3. Clone this repo and install the dependencies:  

	```bash
	cd [ComfyUI_Path]\custom_nodes
	git clone https://github.com/lihaoyun6/ComfyUI-WanVideoWrapper
	pip install -r ComfyUI-WanVideoWrapper\requirements.txt
	```

## Usage
Unlike other attention schemes, draft-attention does not currently support arbitrary resolution video generation.  
So you need to use the **`Find Draft-Attention Bucket`** node to convert the video size you set to a size that draft-attention can handle.  

![image](https://github.com/user-attachments/assets/f9a75df1-4843-4b34-ac9b-24e5f6f5602d)

## Thanks
[ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) - kijai  
[draft-attention](https://github.com/shawnricecake/draft-attention) - shawnricecake  
[Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention) - mit-han-lab  
[Euler/d](https://github.com/eddyhhlure1Eddy/Euler-d) - eddyhhlure1Eddy  
