import numpy as np

IDENTITY = np.array(
            [
                [1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]
            ]
        )

class TransformStack():
    def __init__(self) -> None:
        self._stack = [IDENTITY]

    def push(self, transform: np.array) -> None:
        """
        Receives a 4x4 matrix for transforming every point inside a transform node.
        Stores it, compounded with the latest transform in the stack.
        """
        res = np.matmul(self.peek(), transform)
        self._stack.append(res)

    def pop(self) -> np.array:
        """
        Removes the last transform from the stack, representing the end of a transform node.
        Never removes the identity matrix from the final slot.
        """
        if not np.array_equal(self.peek(), IDENTITY): return self._stack.pop()
        return self.peek()

    def peek(self) -> np.array:
        """
        Returns the latest transform, without altering the stack.
        """
        return self._stack[-1]