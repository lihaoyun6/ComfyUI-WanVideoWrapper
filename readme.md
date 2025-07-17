# ComfyUI-WanVideoWrapper
## This project only supports NVIDIA RTX30/40 and A100/H100 GPU!

This is a modified version of ComfyUI-WanVideoWrapper.  
 I
added **[draft-attention](https://github.com/shawnricecake/draft-attention)** support and **[Euler/d](https://github.com/eddyhhlure1Eddy/Euler-d)** scheduler to ComfyUI-WanVideoWrapper to further speed up video inference.  
And this update significantly improves the inference speed of NVIDIA Ampere / Ada / Hopper GPU.  
The average inference speed is increased by more than 25~40% compared to sage-attn on any Ampere GPU.  
**[[中文版本](./readme_zh.md)]** **[[original readme](./original_readme.md)]**  

## Demo

| Input Image | Sage Attention | Draft Attention (75% sparsity) | Draft Attention (90% sparsity) |
| :----:  | :----: | :----: | :----: |
| <img width=400 src="https://github.com/user-attachments/assets/5c8699d4-c08d-4976-a7a4-7e35a2be4068"> | <video src="https://github.com/user-attachments/assets/d86e6008-37cb-4ae2-ab33-d5b28cc84802"> | <video src="https://github.com/user-attachments/assets/e34c3f54-c2c7-4ec1-bf6f-c91bbfca619a"> | <video src="https://github.com/user-attachments/assets/5ce30f31-1a93-486b-89e1-6f20ae2307b9"> |
| Sampling time: | 70.38s | 48.31s | 40.91s |  

**Prompt:** *"The girl raised her hand to block the sun, with a charming smile on her face, and the camera gradually zoomed out"*  

## Usage
After installing the backend library and ComfyUI plugin, you can use it just like the original WanVideoWrapper.
But unlike other attention mode, draft-attention does not currently support arbitrary resolution video generation.  
So you need to use the **`Find Draft-Attention Bucket`** node to convert the video size you set to a size that draft-attention can handle.  

![image](https://github.com/user-attachments/assets/e1b1245c-e2b5-4d74-be07-abcddaf5b95c)  

## Installation

***! Please uninstall the original Comfyui-WanVideoWrapper before installing my version!***

### Pre-Build Installation (✅Recommend):  
1. Clone this repo and install the dependencies:  

	```bash
	cd [ComfyUI_Path]\custom_nodes
	git clone https://github.com/lihaoyun6/ComfyUI-WanVideoWrapper
	pip install -r ComfyUI-WanVideoWrapper\requirements.txt
	```

2. Install flash-attention and Block-Sparse-Attention.   
	> There are only wheels for `Windows_x64` here!  
	> It's recommended to update GPU driver and install `torch2.7.0+cu128`, then install wheels according to your python version:

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

---

### Manual Installation:
1. Clone this repo and install the dependencies:  

	```bash
	cd [ComfyUI_Path]\custom_nodes
	git clone https://github.com/lihaoyun6/ComfyUI-WanVideoWrapper
	pip install -r ComfyUI-WanVideoWrapper\requirements.txt
	```

2. Build and install Block-Sparse-Attention:
	> Please make sure you have installed the CUDA>=11.6, C++ compiler and PyTorch  

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

3. Install flash-attention:  

	```bash
	pip install flash-attention --no-build-isolation
	```

## Thanks
[ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) - kijai  
[draft-attention](https://github.com/shawnricecake/draft-attention) - shawnricecake  
[Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention) - mit-han-lab  
[Euler/d](https://github.com/eddyhhlure1Eddy/Euler-d) - eddyhhlure1Eddy  
