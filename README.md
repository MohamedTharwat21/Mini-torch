# Mini-torch

## Here is a walk-through of the main classes in `minnn.py`.
0. [np](#np)
0. [Tensor](#Tensor)
0. [Op](#Op)
0. [ComputationGraph](#ComputationGraph)
0. [Initializer](#Initializer)
0. [Model & Trainer](#Model-&-Trainer)

 ![torch cg](https://github.com/user-attachments/assets/ca860c5d-0718-437d-8ab1-ecb0dc4db608)

Computational Grahps in Pytorch


## **np**

**`np`** is an alias for the numerical processing library that i will be using to make computation efficient. By default i use [numpy](https://numpy.org/), a widely used numerical library that you may know of already. For brief tutorials, you can check the links provided at the end of this page. Alternatively, you can use **`cupy`**, an interface that is basically identical to **`numpy`**, but allows computation on the GPU, which can be useful for speed purposes.

- For this project using **`numpy`** with CPU will already be fast enough (around 6s per iter in our running). In my final testing, i would probably use **`numpy`**.  


## **Tensor**

The **`Tensor`** class is a Tensor data structure, with the underlying data stored in a multidimensional array.


![tweaking_nn](https://github.com/user-attachments/assets/237ef635-7d13-4fbb-9499-3feb81155725)


- This class is very similar to **`torch.Tensor`**.
- **`Tensor.data`** is the field that contains the main data for this tensor, this field is a **`np.ndarray`**. The updates of the parameters should be directly changing this data.
- **`Tensor.grad`** is the field for storing the gradient for this tensor. There can be three types of values for this field:
- -> **`None`** : which denotes zero gradient.
- -> **`np.ndarray`** : which should be the same size as the **`Tensor.data`**, denoting dense gradients.
- -> **`Dict[int, np.ndarray]`** : which is a simple simulation of sparse gradients for 2D matrices (embeddings). The key **`int`** denotes the index into the first dimension, while the value is a **`np.ndarray`** which shape is **`Tensor.data.shape[1]`** , denoting the gradient for the column slice according to the index.
- **`Tensor.op`** , which is an **`Op`** (see below) that generates this **`Tensor`**, if **`None`** then mostly not calculated but inputted.
- **`Parameter`** is a simple sub-class of **`Tensor`**, denoting persistent model parameters.

## **Op**

This class implements an operation that is part of a **`ComputationGraph`**.

- **`Op.forward`** and **`Op.backward`**: these are the forward and backward methods calculating the operation itself and its gradient.
- **`Op.ctx`**: this field is a data field that is populated during the **`forward`** operation to store all the relevant values (input, output, intermediate) that must be used in **`backward`** to calculate gradients. I provide two helper methods **`store_ctx`** and **`get_ctx`** to do these, but please feel free to store things in your own way, I will not check **`Op.ctx`**.
- **`Op.full_forward`**: This is a simple wrapper for the actual **`forward`**, adding only one thing to make it convenient, recording the **`Tensor.op`** for the outputted **`Tensor`** so that you do not need to add this in **`forward`**.

## **ComputationGraph**

This class is the one that keeps track of the current computational graph.

![cg_torh](https://github.com/user-attachments/assets/ea466abd-a74f-48b3-a62d-1c6d05eb9f89)


- It simply contains a list of **`Op`**s, which are registered in **`Op.__init__`**.
- In forward, these **`Op`** are appended incrementally in calculation order, and in backward (see function **`backward`**, they are visited in reversed order).

## Initializer

This is simply a collection of initializer methods that produces a **`np.ndarray`** according to the specified shape and other parameters like initializer ranges.

## Model & Trainer

- **`Model`** maintains a collection of **`Parameter`**. I provide **`add_parameters`** as a shortcut of making a new **`Parameter`** and adding it to the model.
- **`Trainer`** takes a **`Model`** and handles the update of the parameters. **`Trainer.update`** denotes one update step which will be implemented in the sub-classes.
- **`SGDTrainer`** is a simple SGD trainer, notice that here i check whether **`Tensor.grad`** is sparse (simulated by a python dictionary) or not, and update accordingly. (In our enviroment with CPU, enabling sparse update is much faster, but not necessarily with GPU).
- Notice that at the end of each **`update`**, I also clear the gradients (clearing by simply setting **`Tensor.grad=None`**). This can usually be two separate steps, but i combine them here for convenience.

## Graph computation algorithms

- **`reset_computation_graph`** discards the previous **`ComputationGraph`** (together with previous **`Op`**s and intermediate **`Tensor`**s) and make a new one. This should be called at the start of each computation loop.
- **`forward`** gets the **`np.ndarray`** value of a **`Tensor`**. Since i calculate everything greedily, this step is simply retriving the **`Tensor.data`**.
- **`backward`** assign a scalar gradient **`alpha`** to a tensor and do backwards according to the reversed order of the **`Op`** list stored inside **`ComputationGraph`**.


## Backpropable functions

![backpropagation](https://github.com/user-attachments/assets/872fd54e-02c5-4402-9e88-b1e00a8a6575)

- The remaining **`Op*`** are all sub-classes of **`Op`** and denotes a specific function. i provide some operations and ask you to implement some of them.
- Take **`OpDropout`** as an example, here i implement the inverted dropout, which scales values by **`1/(1-drop)`** in forward. In **`forward`**, (if training), i obtain a **`mask`** using **`np.random`** and multiply the input by this. All the intermediate values (including input and output) are stored using **`store_ctx`**. In **`backward`**, i obtain the graident of the output **`Tensor`** by retriving previous stored values. Then the calcualted gradients are assigned to the input **`Tensor`** by accumulate_grad.
- Finally, there are some shortcut functions to make it more convenient.




![lookup_table](https://github.com/user-attachments/assets/06eb8fcd-8475-47b7-8202-de3f0d739e74)
