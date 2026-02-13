import os
import sys

import torch

import triton
from triton.backends.compiler import GPUTarget
import triton.compiler
import triton.language as tl

#DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def addvec(x_ptr, y_ptr, output_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < size

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):

    output = torch.empty_like(x)
    size = output.numel()

    grid = lambda x: (triton.cdiv(size, x["BLOCK_SIZE"]), )

    addvec[grid](x, y, output, size, BLOCK_SIZE=1024)

    return output


def run():
    
    torch.manual_seed(0)

    size = 98432

    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)

    triton_out = add(x + y)
    print(triton_out)


def compile(target=None):
    if not target:
        target = GPUTarget("cuda", 100, 32)

    src = triton.compiler.ASTSource(addvec, {"x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32", "size": "u32", "BLOCK_SIZE": "constexpr"}, constexprs={"BLOCK_SIZE": 1024})

    kern = triton.compile(src, target=target)
    return kern


def output_kernels():
    targets = [
        GPUTarget("cuda", 100, 32),
        GPUTarget("hip", "gfx1200", 32),
        GPUTarget("xpu", {"sub_group_sizes": [32], "max_num_sub_groups": 32}, 32),
    ]

    for target in targets:
        k = compile(target)

        out_dir = "./%s" % target.backend
        os.makedirs(out_dir, exist_ok=True)
        
        for k,src in k.asm.items():
            out_path = os.path.join(out_dir, k)
            print("Outputting %s" % out_path) 
            with open(out_path, "wb") as ofd:
                if isinstance(src, str):
                    ofd.write(src.encode("utf8"))
                else:
                    ofd.write(src)


if __name__ == "__main__":
    #import pdb; pdb.set_trace()
    #compiled_vec = triton.jit(addvec)
    #print(compiled_vec)

    #run()
    #kern = compile()
    #import pdb; pdb.set_trace()
    #print(kern)
    output_kernels()
