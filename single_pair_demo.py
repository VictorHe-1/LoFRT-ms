import argparse
from pprint import pprint

import mindspore as ms
import cv2
import matplotlib.cm as cm
import numpy as np

from src.utils.plotting import make_matching_figure


from src.models import LoFTR, default_cfg
from src.utils.timer import Timer


def image_pad(img, pad_shape, coarse_scale=8):
    img_shape = img.shape
    # return  img, np.ones([img_shape[0] // coarse_scale, img_shape[1] // coarse_scale], dtype=np.bool_), 1
    # pad_shape = img_shape  # uncomment this line to disable pad
    img = np.pad(img, ((0, pad_shape[0]-img_shape[0]), (0, pad_shape[1]-img_shape[1])))

    coarse_img_h, coarse_img_w = img_shape[0] // coarse_scale, img_shape[1] // coarse_scale
    coarse_pad_h, coarse_pad_w = pad_shape[0] // coarse_scale, pad_shape[1] // coarse_scale
    mask_c = get_mask([coarse_img_h, coarse_img_w], [coarse_pad_h, coarse_pad_w], coarse_border_thresh=0)
    return img, mask_c


def get_mask(img_shape, pad_shape, coarse_border_thresh=2):
    h, w = img_shape

    mask_c_margin = np.ones(pad_shape, dtype=np.bool_)

    mask_c_margin[:coarse_border_thresh] = False
    mask_c_margin[h-coarse_border_thresh:] = False

    mask_c_margin[:, :coarse_border_thresh] = False
    mask_c_margin[:, w-coarse_border_thresh:] = False

    return mask_c_margin


def infer(args):
    ms.context.set_context(mode=args.mode, device_target=args.device, pynative_synchronize=True)
    # The default config uses dual-softmax.
    # The outdoor and indoor models share the same config.
    # You can change the default values like thr and coarse_match_type.
    with Timer('total'):
        with Timer('build net and load weight'):
            model = LoFTR(config=default_cfg)
            model.set_train(False)
            ckpt_path = "./models/ms_outdoor_ds.ckpt"
            ms.load_checkpoint(ckpt_path, model)

        ms.amp.auto_mixed_precision(network=model, amp_level=args.amp_level)

        with Timer("preprocess"):
            # Load example images
            img0_pth = args.image0
            img1_pth = args.image1
            out_path = args.out_path
            img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
            img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
            img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//16*8, img0_raw.shape[0]//16*8))  # input size shuold be divisible by 8
            img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//16*8, img1_raw.shape[0]//16*8))

            img0_pad, mask_c0 = image_pad(img0_raw, (640, 640))
            img1_pad, mask_c1 = image_pad(img1_raw, (640, 640))

            img0 = ms.Tensor(img0_pad)[None][None] / 255.
            img1 = ms.Tensor(img1_pad)[None][None] / 255.

            mask_c0 = ms.Tensor(mask_c0)[None]
            mask_c1 = ms.Tensor(mask_c1)[None]

        with Timer("inference"):
            # Inference with LoFTR and get prediction
            import numpy as np
            img0 = ms.Tensor(np.load("./debug/image0.npy"))
            img1 = ms.Tensor(np.load("./debug/image0.npy"))
            mask_c0 = ms.Tensor(np.load("./debug/mask0.npy"))
            mask_c1 = ms.Tensor(np.load("./debug/mask1.npy"))
            match_kpts_f0, match_kpts_f1, match_conf, match_masks, _ = model(img0, img1, mask_c0, mask_c1)
        # with Timer("second inference"):
        #     # Inference with LoFTR and get prediction
        #     match_kpts_f0, match_kpts_f1, match_conf, match_masks = model(img0, img1, mask_c0, mask_c1)

        with Timer("from device to host"):
            match_kpts_f0 = match_kpts_f0.squeeze(0).asnumpy()  # (num_max_match, 2)
            match_kpts_f1 = match_kpts_f1.squeeze(0).asnumpy()
            match_conf = match_conf.squeeze(0).asnumpy()  # (num_max_match,)
            match_masks = match_masks.squeeze(0).asnumpy()  # (num_max_match,)

        num_valid_match = match_masks.sum()
        match_kpts_f0 = match_kpts_f0[:num_valid_match]
        match_kpts_f1 = match_kpts_f1[:num_valid_match]
        match_conf = match_conf[:num_valid_match]
        match_masks = match_masks[:num_valid_match]
        breakpoint()
        print(f'matched point num: {num_valid_match}')
    with Timer("draw and save"):
        # Draw
        color = cm.jet(match_conf)
        text = [
            'LoFTR',
            'Matches: {}'.format(len(match_kpts_f0)),
        ]

        make_matching_figure(img0_raw, img1_raw, match_kpts_f0, match_kpts_f1, color, text=text, path=out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Runs loftr inference demo on a image"
        )
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/ms-outdoor_ds.ckpt"
    )

    parser.add_argument(
        "--image0",
        type=str,
        default="assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
    )

    parser.add_argument(
        "--image1",
        type=str,
        default="assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
    )

    parser.add_argument(
        "--out-path",
        type=str,
        default="assets/phototourism_sample_images/paired-united_states_capitol_1.jpg"
    )


    parser.add_argument("--device", type=str, default="Ascend", help="The device to run generation on.")

    parser.add_argument("--mode", type=int, default=1, help="MindSpore context mode. 0 for graph, 1 for pynative.")

    parser.add_argument("--amp_level", type=str, default="O0", help="auto mixed precision level O0, O2.")

    parser.add_argument(
        "--enable-plt-visual",
        action="store_true",
        help=(
            "Button to enable matplot visualization."
        ),
    )

    o_args = parser.parse_args()
    pprint(vars(o_args))
    infer(o_args)
