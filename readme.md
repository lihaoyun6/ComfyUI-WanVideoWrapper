# ComfyUI-WanVideoWrapper
## This project can only be run on Linux or WSL2 on Windows!

This is a modified version of ComfyUI-WanVideoWrapper.  
 I
added **[draft-attention](https://github.com/shawnricecake/draft-attention)** support and **[Euler/d](https://github.com/eddyhhlure1Eddy/Euler-d)** scheduler to ComfyUI-WanVideoWrapper to further speed up video inference.  
And this update significantly improves the inference speed of NVIDIA Ampere GPU (sm80).  
The average inference speed is increased by more than 30% compared to sage-attention.  
**[[中文版本](./readme_zh.md)]** **[[original readme](./original_readme.md)]**  

## Installation

### Pre-Build Installation:
1) This plugin is based on **[Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)** and **[flash-attention](https://github.com/Dao-AILab/flash-attention)**, so you need to install them first:  

```bash
pip install flash-attention
pip install https://github.com/lihaoyun6/ComfyUI-WanVideoWrapper/releases/download/v0.0.1/block_sparse_attn-0.0.1-nv-sm80-cp313-cp313-linux_x86_64.whl
```

> This wheel only supports NVIDIA Ampere GPU (sm80) and Python 3.13! If you need to use it on other GPUs, please build it yourself.  
> 

2) Then clone this repo into `custom_nodes` folder.  

3) Install dependencies: `pip install -r requirements.txt`
   or if you use the portable install, run this in `ComfyUI_windows_portable` folder:

```bash
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-WanVideoWrapper\requirements.txt
```

===

### Manual Installation:
1) Build and install Block-Sparse-Attention:

```bash
pip install packaging
#Install ninja for faster compilation
pip install ninja
#If your system memory is less than 96GB
export MAX_JOBS=5
#Download source code and install.
#This process may take more than an hour depending on your CPU and memory. Please be patient.
git clone https://github.com/mit-han-lab/Block-Sparse-Attention
cd Block-Sparse-Attention
python setup.py install
```
2) Install flash-attention:  

```bash
pip install flash-attention
```
3) Then clone this repo into `custom_nodes` folder.  

4) Install dependencies: `pip install -r requirements.txt`
   or if you use the portable install, run this in `ComfyUI_windows_portable` folder:

```bash
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-WanVideoWrapper\requirements.txt
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
