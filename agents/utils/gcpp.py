import torch


def retrieve_buffer(buffer, batch_size, learned_classes, exclude_id):
    """
    Sample replay examples from buffer across learned classes (excluding one class).

    Args:
        buffer (dict): Dictionary mapping class_id -> tensor of stored samples.
        batch_size (int): Total number of replay samples to collect.
        learned_classes (list or iterable): List of class IDs that have been learned.
        exclude_id (int): The class ID to exclude from replay sampling.

    Returns:
        torch.Tensor: Concatenated replay samples (or None if no samples available).
    """

    x_buff = None
    per_class = max(1, batch_size // len(learned_classes))

    for cl_id in learned_classes:
        if cl_id != exclude_id and cl_id in buffer:
            rand_idxs = torch.randperm(buffer[cl_id].size(0))[:per_class]
            x_buff_cl_id = buffer[cl_id][rand_idxs]

            if x_buff is not None:
                x_buff = torch.cat([x_buff, x_buff_cl_id], dim=0)
            else:
                x_buff = x_buff_cl_id

    return x_buff


def update_buffer(buffer, class_id, x_train_id, max_mem_per_class):
    """
    Update the replay buffer for a given class.

    Args:
        buffer (dict): Dictionary mapping class_id -> tensor of stored samples.
        class_id (int): The class ID to update.
        x_train_id (array-like or tensor): Training samples for the class.
        max_mem_per_class (int): Maximum number of samples to store per class.

    Returns:
        dict: Updated buffer.
    """

    n_buff = min(max_mem_per_class, len(x_train_id))
    buffer[class_id] = torch.tensor(x_train_id[:n_buff])

    return buffer
