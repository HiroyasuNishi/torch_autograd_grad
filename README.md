# torch.autograd.gradの引数についての説明
引数を以下の4つに分けて説明する.

- a scalar
- scalars (scalarが要素のタプル)
- a tensor
- tensors (tensorが要素のタプル)



## 基本的に
grad = torch.autograd.grad(outputs, inputs, grad_outputs)とする
grad_outputsとoutputsのtensors or scalarsのshapeと個数は基本的に同じ.
gradのtensors or scalarsのshapeと個数は基本的にinputsと同じ. inputsがa tensorの場合は、gradは要素が1つのtuple
つまり、outputsとgrad_outputsが対応していて、inputsとgradが対応している.

## outputsとinputsとgrad_outputsの形状に応じて場合分けする

以下はそれぞれのcaseのコード例に共通のコード

```python
import torch

a = torch.tensor([2.0, 3.0], requires_grad=True)
b = torch.tensor([5.0, 7.0], requires_grad=True)
m = torch.tensor([[1.0, 3.0],
                  [2.0, 5.0]])
m2 = torch.tensor([[2.0, 6.0],
                  [2.0, 1.0]])
x = m@a + m2@b
x2 = m@a - m2@b
y = x.sum()
y2 = x2.sum()
```

### case 1-1-1. outputs: a scalar. inputs: a tensor. grad_outputs: None.
つまり, autograd.grad(y, a, grad_outputs=None)の計算  
(返ってくる値は)-> (∂y/∂a, )

```python
y_grad = torch.autograd.grad(y, a)
print(y_grad)
```

### case 1-1-2. outputs: a scalar. inputs: a tensor. grad_outputs: a scalar (with the same shape and the same number of tensor or scalar as outputs) (outputsとtensor or scalarのshapeと個数が同じ.)
autograd.grad(y, a, grad_outputs=v)の計算  
-> (∂y/∂a * v, )

```python
y_grad = torch.autograd.grad(y, a, grad_outputs=torch.tensor(2))
print(y_grad)
```

### case 1-2-1. outputs: a scalar. inputs: tensors. grad_outputs: None
autograd.grad(y, (a1, a2), grad_outputs=None)の計算  
-> (∂y/∂a1, ∂y/∂a2)

```python
y_grad = torch.autograd.grad(y, (a, b))
print(y_grad)
```

### case 1-2-2. outputs: a scalar. inputs: tensors. grad_outputs: a scalar (with the same shape and the same number of tensor or scalar as outputs)
autograd.grad(y, (a1, a2), grad_outputs=v)の計算  
-> (∂y/∂a1 * v1, ∂y/∂a2 * v1)

```python
y_grad = torch.autograd.grad(y, (a, b), grad_outputs=(torch.tensor(2),))
print(y_grad)
```

### case 2-1-1. outputs: scalars. inputs: a tensor. grad_outputs: None.
autograd.grad((y1, y2), a, grad_outputs=None)の計算  
-> (∂y1/∂a + ∂y2/∂a, )  
つまり, autograd.grad(y1+y2, a, grad_outputs=None)と同じ

```python
y_grad = torch.autograd.grad((y, y2), a)
print(y_grad)
y_grad = torch.autograd.grad(y+y2, a)
print(y_grad)
```

### case 2-1-2. outputs: scalars. inputs: a tensor. grad_outputs: scalars
autograd.grad((y1, y2), a, grad_outputs=(v1, v2))の計算  
-> (∂y1/∂a*v1 + ∂y2/∂a*v2, )  
つまり, autograd.grad(y1*v1 + y2*v2, a, grad_outputs=None)と同じ

```python
y_grad = torch.autograd.grad((y, y2), a, grad_outputs=(torch.tensor(2), torch.tensor(3)))
print(y_grad)
```

### case 2-2-1. outputs: scalars. inputs: tensors. grad_outputs: None.
autograd.grad((y1, y2), (a1, a2), grad_outputs=None)の計算  
-> (∂y1/∂a1 + ∂y2/∂a1, ∂y1/∂a2 + ∂y2/∂a2)

```python
y_grad = torch.autograd.grad((y, y2), (a, b))
print(y_grad)
```

### case 2-2-2. outputs: scalars. inputs: tensors. grad_outputs: scalars (with the same shape and the same number of tensor or scalar as outputs).
autograd.grad((y1, y2), (a1, a2), grad_outputs=(v1, v2))の計算  
-> (∂y1/∂a1*v1 + ∂y2/∂a1*v2, ∂y1/∂a2*v1 + ∂y2/∂a2*v2)

```python
y_grad = torch.autograd.grad((y, y2), (a, b), grad_outputs=(torch.tensor(2), torch.tensor(3)))
print(y_grad)
```

### case 3-1-1. outputs: a tensor. inputs: a tensor. grad_outputs: None
エラー. grad_outputsを指定する必要あり

### case 3-1-2. outputs: a tensor. inputs: a tensor. grad_outputs: a tensor (with the same shape and the same number of tensor or scalar as outputs).
autograd.grad(x, a, grad_outputs=v)の計算  
-> ((∂x/∂a)^T * v, )
[vector-jacobian product](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)を参考

```python
y_grad = torch.autograd.grad(x, a, grad_outputs=b)
print(y_grad)
```

### case 3-2-1. outputs: a tensor. inputs: tensors. grad_outputs: None
エラー. grad_outputsを指定する必要あり

### case 3-2-2. outputs: a tensor. inputs: tensors. grad_outputs: a tensor (with the same shape and the same number of tensor or scalar as outputs).
autograd.grad(x, (a1, a2), grad_outputs=v)の計算  
-> ((∂x/∂a1)^T * v, (∂x/∂a2)^T * v)

```python
y_grad = torch.autograd.grad(x, (a, b), grad_outputs=(torch.tensor([2, 3]),))
print(y_grad)
```

### case 4-1-1. outputs: tensors. inputs: a tensor. grad_outputs: None
エラー. grad_outputsを指定する必要あり  

### case 4-1-2. outputs: tensors. inputs: a tensor. grad_outputs: tensors (with the same shape and the same number of tensor or scalar as outputs).
autograd.grad((x1, x2), a, grad_outputs=(v1, v2))の計算  
-> ((∂x1/∂a)^T * v1 + (∂x2/∂a)^T * v2, )

```python
y_grad = torch.autograd.grad((x, x2), b, grad_outputs=(torch.tensor([1, 1]), torch.tensor([1, 2])))
print(y_grad)
```

### case 4-2-1. outputs: tensors. inputs: tensors. grad_outputs: None
エラー. grad_outputsを指定する必要あり

### case 4-2-2. outputs: tensors. inputs: tensors. grad_outputs: tensors (with the same shape and the same number of tensor or scalar as outputs).
autograd.grad((x1, x2), a, grad_outputs=(v1, v2))の計算  
-> ((∂x1/∂a1)^T * v1 + (∂x2/∂a1)^T * v2, (∂x1/∂a2)^T * v1 + (∂x2/∂a2)^T * v2)

```python
y_grad = torch.autograd.grad((x, x2), (a, b), grad_outputs=(torch.tensor([1, 1]), torch.tensor([1, 2])))
print(y_grad)
```