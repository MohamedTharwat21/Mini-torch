

## Tensor

The **`Tensor`** class is a Tensor data structure, with the underlying data stored in a multidimensional array.

- This class is very similar to **`torch.Tensor`**.
- **`Tensor.data`** is the field that contains the main data for this tensor, this field is a **`np.ndarray`**. The updates of the parameters should be directly changing this data.
- **`Tensor.grad`** is the field for storing the gradient for this tensor. There can be three types of values for this field:
- -> **`None`** : which denotes zero gradient.
- -> **`np.ndarray`** : which should be the same size as the **`Tensor.data`**, denoting dense gradients.
- -> **`Dict[int, np.ndarray]`** : which is a simple simulation of sparse gradients for 2D matrices (embeddings). The key **`int`** denotes the index into the first dimension, while the value is a **`np.ndarray`** which shape is **`Tensor.data.shape[1]`** , denoting the gradient for the column slice according to the index.
- **`Tensor.op`** , which is an **`Op`** (see below) that generates this **`Tensor`**, if **`None`** then mostly not calculated but inputted.
- **`Parameter`** is a simple sub-class of **`Tensor`**, denoting persistent model parameters.

