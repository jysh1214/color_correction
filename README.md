# COLOR CORRECTION

It's a implement of [Contrast Enhancement of Brightness-Distorted Images by Improved Adaptive Gamma Correction](https://arxiv.org/abs/1709.04427), writed by c++/CUDA.

## Env
- opencv
- clang
- CUDA (option)

## OS
- ubuntu
- mac (don't support CUDA)

## Compile & Use
```
make clean && make
./color_correction <YOUR_IMAGE>
```

## With CUDA
change 4th line of the `makefile`
```
CUDA=1
```

## Result
<Table>
    <tr>
        <th>Input Image</th>
        <th>Output Image</th>
    </tr>
    <tr>
        <td><img src="https://github.com/jysh1214/color_correction/blob/main/asset/input.jpg" alt="Original" width="400" height="500" align="middle"/></td>
        <td><img src="https://github.com/jysh1214/color_correction/blob/main/asset/output.png" alt="Original" width="400" height="500" align="middle"/></td>
    </tr>
</Table>

## Reference
- [https://github.com/leowang7/iagcwd](https://github.com/leowang7/iagcwd)