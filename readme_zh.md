# ComfyUI-WanVideoWrapper
## 此插件仅支持Linux系统或在Windows中通过WSL2运行!

本项目是修改版的`ComfyUI-WanVideoWrapper`.  
我添加了 **[draft-attention](https://github.com/shawnricecake/draft-attention)** 注意力方案和 **[Euler/d](https://github.com/eddyhhlure1Eddy/Euler-d)** 调度器支持以进一步提升推理速度.  
此更新显著的提高了 NVIDIA Ampere GPU (sm80) 上的视频生成速度,  
 相比于 sage-attention 2.2.0 , 使用 draft-attention 平均可提速30%以上.  
**[[中文版本](./readme_zh.md)]** **[[original readme](./original_readme.md)]**  

## 安装

### 预构建安装:
1) 本插件基于 **[Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)** 和 **[flash-attention](https://github.com/Dao-AILab/flash-attention)**, 所以你需要先安装齐备:  

```bash
pip install flash-attention
pip install https://github.com/lihaoyun6/ComfyUI-WanVideoWrapper/releases/download/v0.0.1/block_sparse_attn-0.0.1-nv-sm80-cp313-cp313-linux_x86_64.whl
```

> 此预构建 wheel 仅支持 NVIDIA Ampere GPU (sm80) 和 Python 3.13! 如果你需要配合其他架构的 GPU 使用, 请选择`编译安装`方案.  
> 

2) 克隆此项目到 ComfyUI 的 `custom_nodes` 目录中.  

3) 然后使用此命令安装依赖项: `pip install -r requirements.txt`
   或者如果你使用的是便携版 ComfyUI, 则请在 `ComfyUI_windows_portable` 文件夹中执行:

```bash
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-WanVideoWrapper\requirements.txt
```

===

### 编译安装:
1) 编译并安装 Block-Sparse-Attention:

```bash
#安装 ninja 可以有效提高 CUDA 编译效率
pip install ninja
#如果你的系统内存不足 96GB, 请限制最大编译线程数为 5
export MAX_JOBS=5
#下载源代码并开始编译安装.
#此过程根据你的 CPU 和内存容量, 可能会持续一个小时甚至更久, 请耐心等待安装完成.
git clone https://github.com/mit-han-lab/Block-Sparse-Attention
cd Block-Sparse-Attention
python setup.py install
```
2) 安装 flash-attention:  

```bash
pip install flash-attention
```
3) 克隆此项目到 ComfyUI 的 `custom_nodes` 目录中.  

4) 然后使用此命令安装依赖项: `pip install -r requirements.txt`
   或者如果你使用的是便携版 ComfyUI, 则请在 `ComfyUI_windows_portable` 文件夹中执行:

```bash
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-WanVideoWrapper\requirements.txt
```

## Usage
与其他注意力方案不同, draft-attention 目前暂未支持任意分辨率视频生成.  
所以在搭建工作流时请务必使用插件提供的 **`Find Draft-Attention Bucket`** 节点来将你输入的分辨率转换为 draft-attention 能接受的格式.  

![image](foo-bar)

## 鸣谢
[ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) - kijai  
[draft-attention](https://github.com/shawnricecake/draft-attention) - shawnricecake  
[Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention) - mit-han-lab  
[Euler/d](https://github.com/eddyhhlure1Eddy/Euler-d) - eddyhhlure1Eddy  
