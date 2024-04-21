from torch import Tensor


def label_segmentations(trimap) -> Tensor:
    """
    Convert a float trimap ({1, 2, 3} / 255.0) into a float tensor with
    pixel values in the range 0.0 to 1.0 so that the border pixels
    can be properly displayed.
    """

    x = (trimap * 255.0 - 1) / 2
    # Change all 0.5 values to 1 (background and border become the same class)
    x[x == 0.5] = 1

    return x
