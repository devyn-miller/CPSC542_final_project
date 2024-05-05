import numpy as np

def combine_lab_channels(l_channel, ab_channels):
    """
    Combine the L channel (lightness) with the AB channels (color) to create a full LAB image.

    Args:
        l_channel (np.ndarray): The L channel (lightness) of the LAB image.
        ab_channels (np.ndarray): The AB channels (color) of the LAB image.

    Returns:
        np.ndarray: The full LAB image.
    """
    # Ensure input shapes are correct
    assert l_channel.shape[-1] == 1, "L channel should have shape (x, x, 1)"
    assert ab_channels.shape[-1] == 2, "AB channels should have shape (x, x, 2)"

    # Combine the channels
    lab_image = np.concatenate((l_channel, ab_channels), axis=-1)
    return lab_image
                        