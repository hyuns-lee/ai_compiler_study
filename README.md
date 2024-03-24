# ai_compiler_study
## 1. Setup
- 주어진 TransformerEngine의 **b8eea8a** 브랜치와 동일한 **apply_rotary_pos_emb** 함수를 갖는 버전을 사용하기 위해 다음과 같이 설정
- conda/pip를 이용하여 설정하여 사용했으나 버전과 code dependency로 인해 docker 이미지도 추가로 만듦
### 1.1. conda 사용
- 사용한 conda env에 대한 yaml 파일: environment.yaml
- TransformerEngine을 install 하기 이전에 torch가 설치되어 있어야 설치 가능
- 실제로 설정한 순서는 다음과 같음
    ```shell
    # CUDA 11.8 & cudnn 8.9.7 installed in advance
    conda create -n study python=3.10
    conda activate study
    
    # torch install
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

    # TransformerEngine install
    git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git
    cd TransformerEngine
    export NVTE_FRAMEWORK=pytorch # Optionally set framework
    pip install . # Build and install
    ```
- TransformerEngine stable branch에 포함된 flash-attn 버전(2.4.2) 문제로 flash attention fused kernel이 잘 안불려져서 다음 코드 실행시에 동반되는 flash-attn import 하는 부분은 주석처리함
    ```python
    from transformer_engine.pytorch.attention import (
        RotaryPositionEmbedding,
        apply_rotary_pos_emb,
    )
    ```
### 1.2. docker 사용
- docker image
    ```shell
    docker push hslee55/rope_triton:latest
    ```
- test file path
    - **/root/TransformerEngine/tests/pytorch/test_study.py**


## 2. 결과물
### 2.1. [**이곳에 나오는 pytorch 함수**](https://github.com/NVIDIA/TransformerEngine/blob/b8eea8aaa94bb566c3a12384eda064bda8ac4fd7/transformer_engine/pytorch/attention.py#L1170-L1230)가 실행가능 한 코드

### 2.2. Triton으로 짠 `rope_fw` & `rope_bw` 코드

### 2.3. `rope`의 Pytorch 함수와 Triton 함수의 forward, backward 결과 값의 같다 (`torch.testing.assert_close`)는 것을 보여주는 코드

### 2.4. Profiling한 결과와 개선한 점에 대한 설명
