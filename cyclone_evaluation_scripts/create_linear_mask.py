import numpy as np
import random
import pickle


def generate_masks_and_save(data_list, save_path="./postprocess_mask/mask_dict.pkl"):
    """
    generate Mask and save to pkl file

    Args:
        data_list: list, each element is (id, base_radius, ext_radius)
        save_path: save path
    """

    # initialize parameters
    img_size = 241
    center = img_size // 2  # 120

    # generate grid coordinates (y, x), shape: (img_size, img_size)
    Y, X = np.ogrid[:img_size, :img_size]
    # calculate distance matrix to center, shape: (img_size, img_size)
    dist_map = np.sqrt((X - center) ** 2 + (Y - center) ** 2)

    mask_dict = {}

    print(f"start processing {len(data_list)} Mask...")

    for item in data_list:
        mask_id = item["id"]
        r_base = item["base_radius"]
        r_ext = item["ext_radius"]

        # calculate k and b for linear decay function
        # only calculate k and b when r_ext > 0, to avoid division by zero error
        if r_ext > 0:
            k = 1.0 / r_ext
            b = random.uniform(0, k)
        else:
            k = 0
            b = 0

        # initialize current Mask with zeros
        mask = np.zeros((img_size, img_size), dtype=np.float32)

        # fill base radius (fill 1)
        # logic: distance <= base radius
        mask[dist_map <= r_base] = 1.0

        # process ext radius (linear decay)
        # logic: base radius < distance <= base radius + ext radius
        ext_zone = (dist_map > r_base) & (dist_map <= (r_base + r_ext))

        if r_ext > 0 and np.any(ext_zone):
            # get pixel distance in ext zone
            d_vals = dist_map[ext_zone]

            # x is distance from base radius boundary
            x = d_vals - r_base

            # calculate decay value: Value = 1 - (kx + b)
            # logic: Value decreases as distance x increases
            decay_values = 1.0 - (k * x + b)

            # clip negative values to 0 (although k*r_ext = 1, 1-(1+b) = -b)
            decay_values = np.clip(decay_values, 0, 1.0)

            # assign decay values to ext zone mask
            mask[ext_zone] = decay_values

        # store to dictionary
        mask_dict[mask_id] = mask

    # save to pkl file
    with open(save_path, "wb") as f:
        pickle.dump(mask_dict, f)

    print(f"processing completed！dictionary saved to: {save_path}")
    print(f"contains ids: {list(mask_dict.keys())}")


if __name__ == "__main__":
    input_data = []

    # range(start, stop, step)
    for hour in range(6, 121, 6):
        input_data.append({"id": hour, "base_radius": hour, "ext_radius": 24})

    generate_masks_and_save(input_data)
