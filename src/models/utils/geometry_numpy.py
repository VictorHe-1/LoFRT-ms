import numpy as np


def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).

    Args:
        kpts0 (np.ndarray): [N, L, 2] - <x, y>,
        depth0 (np.ndarray): [N, H, W],
        depth1 (np.ndarray): [N, H, W],
        T_0to1 (np.ndarray): [N, 3, 4],
        K0 (np.ndarray): [N, 3, 3],
        K1 (np.ndarray): [N, 3, 3],
    Returns:
        calculable_mask (np.ndarray): [N, L]
        warped_keypoints0 (np.ndarray): [N, L, 2] <x0_hat, y1_hat>
    """
    kpts0_long = np.round(kpts0).astype(np.int64)

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = np.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], axis=0
    )  # (N, L)
    nonzero_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = np.concatenate([kpts0, np.ones_like(kpts0[:, :, [0]])], axis=-1) * kpts0_depth[..., None]  # (N, L, 3)
    kpts0_cam = np.linalg.inv(K0) @ np.swapaxes(kpts0_h, 2, 1)  # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = np.swapaxes((K1 @ w_kpts0_cam), 2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4)  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = ((w_kpts0[:, :, 0] > 0).astype(np.int32) *
                      (w_kpts0[:, :, 0] < w - 1).astype(np.int32) *
                      (w_kpts0[:, :, 1] > 0).astype(np.int32) *
                      (w_kpts0[:, :, 1] < h - 1).astype(np.int32))
    w_kpts0_long = w_kpts0.astype(np.int64)
    w_kpts0_long[~covisible_mask.astype(bool), :] = 0

    w_kpts0_depth = np.stack(
        [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], axis=0
    )  # (N, L)
    consistent_mask = np.abs((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth) < 0.2
    valid_mask = (nonzero_mask.astype(np.int32) * covisible_mask * consistent_mask.astype(np.int32))

    return valid_mask, w_kpts0
