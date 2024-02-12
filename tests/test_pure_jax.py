import jax
import jax.numpy as jnp

def hermitian(matrix):
    return jnp.conj(matrix.T)

def is_hermitian(matrix):
    return jnp.allclose(matrix, hermitian(matrix))

# Example usage
matrix = jnp.array([[1, 2], [2, 1]])
print(is_hermitian(matrix))  # Output: True

matrix = jnp.array([[1, 2], [3, 4]])
print(is_hermitian(matrix))  # Output: False