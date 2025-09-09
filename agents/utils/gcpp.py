import torch
from utils.data import extract_samples_according_to_labels


def retrieve_buffer(buffer, batch_size, learned_classes):
    x_buff, y_buff = None, None
    per_class = max(1, batch_size // len(learned_classes))

    for cl_id in learned_classes:
        if cl_id in buffer:
            x_class, y_class = buffer[cl_id]
            rand_idxs = torch.randperm(x_class.size(0))[:per_class]

            x_sample = x_class[rand_idxs]
            y_sample = y_class[rand_idxs]

            if x_buff is not None:
                x_buff = torch.cat([x_buff, x_sample], dim=0)
                y_buff = torch.cat([y_buff, y_sample], dim=0)
            else:
                x_buff = x_sample
                y_buff = y_sample

    return x_buff, y_buff


def update_buffer(buffer, class_ids, x_train, y_train, max_mem_per_class):
    for class_id in class_ids:
        (x_train_id, y_train_id) = extract_samples_according_to_labels(
            x_train, y_train, [id]
        )
        n_buff = min(max_mem_per_class, len(x_train_id))
        buffer[class_id] = (
            torch.tensor(x_train_id[:n_buff], dtype=torch.float32),
            torch.tensor(y_train_id[:n_buff], dtype=torch.long),
        )

    return buffer
