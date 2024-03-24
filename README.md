# ai_compiler_study
## 1. Setup
- 주어진 TransformerEngine의 `b8eea8a` 브랜치와 동일한 `apply_rotary_pos_emb` 함수를 갖는 버전을 사용하기 위해 다음과 같이 설정
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
```shell
# Run the test code
python test_study.py
# pytest the code
pytest test_study.py
```
### 2.1. [**이곳에 나오는 pytorch 함수**](https://github.com/NVIDIA/TransformerEngine/blob/b8eea8aaa94bb566c3a12384eda064bda8ac4fd7/transformer_engine/pytorch/attention.py#L1170-L1230)가 실행가능 한 코드
- `test_study.py` 내부의 test_fused_rope 함수 안에 apply_rotary_pos_emb의 fused, unfused 실행이 가능한 코드가 있고 둘 사이의 결과가 같은 것을 확인함
    ```python
    print("[TEST] apply_rotary_pos_emb unfused")
    output_unfused = apply_rotary_pos_emb(
        t, emb, tensor_format=tensor_format, fused=False
    )
    loss_unfused = loss_func(output_unfused)
    loss_unfused.backward()
    grad_unfused = t.grad.detach().clone()
    t.grad = None

    print("[TEST] apply_rotary_pos_emb fused")
    output_fused = apply_rotary_pos_emb(
        t, emb, tensor_format=tensor_format, fused=True
    )
    loss_fused = loss_func(output_fused)
    loss_fused.backward()
    grad_fused = t.grad.detach().clone()
    t.grad = None

    print("[TEST] Check if fused/unfused apply_rotary_pos_emb are same:", end=" ")
    torch.testing.assert_close(output_unfused, output_fused, **get_tol(dtype))
    torch.testing.assert_close(grad_unfused, grad_fused, **get_tol(dtype))
    print("PASS")
    ```

### 2.2. Triton으로 짠 `rope_fw` & `rope_bw` 코드
- `test_study.py` 내부에 triton으로 짠 `rope_fw`, `rope_bw`를 작성하였고, gradient를 쉽게 구하기 위해 `TritonRoPEFunc(torch.autograd.Function)` 안에 넣어서 사용

### 2.3. `rope`의 Pytorch 함수와 Triton 함수의 forward, backward 결과 값의 같다 (`torch.testing.assert_close`)는 것을 보여주는 코드
- `test_study.py`의 `test_fused_rope` 안에 triton으로 작성한 rope인 `TritonRoPEFunc`의 결과와 unfused `apply_rotary_pos_emb`의 결과를 비교함
    ```python
    print("[TEST] apply_rotary_pos_emb with triton")
    out_triton = TritonRoPEFunc.apply(t, emb, tensor_format, None)
    loss_triton = loss_func(out_triton)
    loss_triton.backward()
    grad_triton = t.grad.detach().clone()
    t.grad = None

    print("[TEST] Check if the result of triton RoPE and unfused apply_rotary_pos_emb are same:", end=" ")
    torch.testing.assert_close(out_triton, output_unfused, **get_tol(dtype))
    torch.testing.assert_close(grad_triton, grad_unfused, **get_tol(dtype))
    print("PASS")
    ```

### 2.4. Profiling한 결과와 개선한 점에 대한 설명

## 3. 코드 설명
### 3.1. TritonRoPEFunc
- 주어진 `apply_rotary_pos_emb`가 `fused: True`인 경우 사용하는 torch의 autograd 기능을 동일하게 사용하기 위해 **transformer_engine/pytorch/attention.py**의 `FusedRoPEFunc` class를 참고하여 `TritonRoPEFunc`를 구현함
- `TritonRoPEFunc` class 내부에는 @staticmethod와 함께 forward, backward를 구현함
- forward에서는 triton kernel인 rope_fw를 부르고, backward에서는 triton kernel인 rope_bw를 불러 각각의 기능을 수행함
- tensor_format에 따라서 각각 맞는 형태로 변환하여 triton kernel을 부르고, 그 결과 나오는 출력도 맞는 형태로 변환하여 리턴해줌

### 3.2. rope_fw
- `BLOCK_SIZE`는 `rotary_percent`가 반영된 `hidden_dim`으로 즉, `hidden_dim*rotary_percent`으로 설정하였음
    ```python
    @triton.jit
    def rope_fw(t_ptr, emb_ptr, out_ptr, seq_length, batch_size, head_num, hidden_size, BLOCK_SIZE:tl.constexpr):
        # 1개의 program이 1개의 input row를 담당
        pid = tl.program_id(axis=0)

        # input t의 width가 hidden_size
        t_start = pid * hidden_size

        # emb는 pid가 batch_size*head_num 개 증가할 때마다 업데이트
        # emb의 Width == BLOCK_SIZE
        emb_start = pid//(batch_size*head_num) * BLOCK_SIZE

        # 밑에서 한번에 사용할 input row의 폭이 BLOCK_SIZE//2 이므로 동일하게 emboffset도 BLOCK_SIZE//2만 load
        # emb 연산 시 RotaryPositionEmbedding.forward에서 같은 freqs를 concat하므로 BLOCK_SIZE//2를 load하고 두번 반복해서 사용 가능
        emboffset = emb_start + tl.arange(0, BLOCK_SIZE//2)
        maskemb = emboffset < seq_length*BLOCK_SIZE
        emb = tl.load(emb_ptr + emboffset, mask=maskemb)
        cos_emb = tl.cos(emb)
        sin_emb = tl.sin(emb)

        # input t 에 대해서 left, right로 둘로 나누고, 각각에 대해서 cos_emb, sin_emb에 맞는 elemwise mult.를 해준다
        t_left_offset = t_start + tl.arange(0, BLOCK_SIZE//2)
        t_right_offset = t_start + tl.arange(0, BLOCK_SIZE//2) + BLOCK_SIZE//2

        maskl = t_left_offset < seq_length*batch_size*head_num*hidden_size
        t_left = tl.load(t_ptr + t_left_offset, mask=maskl)
        maskr = t_right_offset < seq_length*batch_size*head_num*hidden_size
        t_right = tl.load(t_ptr + t_right_offset, mask=maskr)

        # output = ⌈cos_emb -sin_emb⌉ ⌈t_left ⌉
        #          ⌊sin_emb cos_emb ⌋ ⌊t_right⌋
        output = t_left * cos_emb - t_right * sin_emb
        tl.store(out_ptr+t_left_offset, output, mask = maskl)

        output =  t_left * sin_emb + t_right * cos_emb
        tl.store(out_ptr+t_right_offset, output, mask = maskr)
    ```

### 3.3. rope_bw
- `BLOCK_SIZE`는 `rope_fw`와 동일하게 `rotary_percent`가 반영된 `hidden_dim`으로 즉, `hidden_dim*rotary_percent`으로 설정하였음
- Forward에서 `M_rot(emb) * input(t) = output` 을 통해서 rotation을 수행했고 backward에서는 `output_grad`를 입력으로 하여 `input_grad`를 구해야 하므로 M_rot<sup>-1</sup> * output_grad = input_grad를 연산해야 함
- M_rot<sup>-1</sup> 은 반대 방향으로 rotation을 하면 되기 때문에 다음의 형태로 변환을 해준다
$$
M_{rot}^{-1} =
\begin{pmatrix}
\cos_{emb} & \sin_{emb} \\
-\sin_{emb} & \cos_{emb}
\end{pmatrix}$$

- 
    ```python
    @triton.jit
    def rope_bw(t_ptr, emb_ptr, out_ptr, seq_length, batch_size, head_num, hidden_size, BLOCK_SIZE:tl.constexpr):
        pid = tl.program_id(axis=0)

        t_start = pid * hidden_size
        emb_start = pid//(batch_size*head_num) * BLOCK_SIZE

        emboffset = emb_start + tl.arange(0, BLOCK_SIZE//2)
        maskemb = emboffset < seq_length*BLOCK_SIZE
        emb = tl.load(emb_ptr + emboffset, mask=maskemb)
        cos_emb = tl.cos(emb)
        sin_emb = tl.sin(emb)

        t_left_offset = t_start + tl.arange(0, BLOCK_SIZE//2)
        t_right_offset = t_start + tl.arange(0, BLOCK_SIZE//2) + BLOCK_SIZE//2

        maskl = t_left_offset < seq_length*batch_size*head_num*hidden_size
        t_left = tl.load(t_ptr + t_left_offset, mask=maskl)
        maskr = t_right_offset < seq_length*batch_size*head_num*hidden_size
        t_right = tl.load(t_ptr + t_right_offset, mask=maskr)

        # output = ⌈cos_emb  sin_emb⌉ ⌈t_left ⌉
        #          ⌊-sin_emb cos_emb⌋ ⌊t_right⌋
        output = t_left * cos_emb + t_right * sin_emb
        tl.store(out_ptr+t_left_offset, output, mask = maskl)

        output =  -t_left * sin_emb + t_right * cos_emb
        tl.store(out_ptr+t_right_offset, output, mask = maskr)
    ```