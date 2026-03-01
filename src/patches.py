import time
import torch
import numpy as np

def apply_patches():
    # --- FIX 1: craft-text-detector compatibility with newer torchvision ---
    try:
        import torchvision.models.vgg as vgg
        if not hasattr(vgg, 'model_urls'):
            vgg.model_urls = {
                'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
            }
        elif 'vgg16_bn' not in vgg.model_urls:
             vgg.model_urls['vgg16_bn'] = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
    except ImportError:
        pass

    # --- FIX 2: craft-text-detector compatibility with newer NumPy (ragged arrays) ---
    try:
        import craft_text_detector.craft_utils as craft_utils
        import craft_text_detector.predict as predict
        import craft_text_detector.image_utils as image_utils
        import craft_text_detector.torch_utils as torch_utils
        import cv2

        def patched_adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
            if len(polys) > 0:
                for k in range(len(polys)):
                    if polys[k] is not None:
                        polys[k] = np.array(polys[k]) * (ratio_w * ratio_net, ratio_h * ratio_net)
            return polys
        
        craft_utils.adjustResultCoordinates = patched_adjustResultCoordinates

        def exhaustive_patched_get_prediction(image, craft_net, refine_net=None, text_threshold=0.7, link_threshold=0.4, low_text=0.4, cuda=False, long_size=1280, poly=True):
            t0 = time.time()
            image = image_utils.read_image(image)
            img_resized, target_ratio, size_heatmap = image_utils.resize_aspect_ratio(image, long_size, interpolation=cv2.INTER_LINEAR)
            ratio_h = ratio_w = 1 / target_ratio
            resize_time = time.time() - t0
            t0 = time.time()
            x = image_utils.normalizeMeanVariance(img_resized)
            x = torch_utils.from_numpy(x).permute(2, 0, 1)
            x = torch_utils.Variable(x.unsqueeze(0))
            if cuda: x = x.cuda()
            preprocessing_time = time.time() - t0
            t0 = time.time()
            with torch_utils.no_grad():
                y, feature = craft_net(x)
            craftnet_time = time.time() - t0
            t0 = time.time()
            score_text = y[0, :, :, 0].cpu().data.numpy()
            score_link = y[0, :, :, 1].cpu().data.numpy()
            if refine_net is not None:
                with torch_utils.no_grad():
                    y_refiner = refine_net(y, feature)
                score_link = y_refiner[0, :, :, 0].cpu().data.numpy()
            refinenet_time = time.time() - t0
            t0 = time.time()
            boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
            boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
            polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
            for k in range(len(polys)):
                if polys[k] is None: polys[k] = boxes[k]
            img_height, img_width = image.shape[:2]
            boxes_as_ratio = [box / [img_width, img_height] for box in boxes]
            polys_as_ratio = [poly / [img_width, img_height] for poly in polys]
            text_score_heatmap = image_utils.cvt2HeatmapImg(score_text)
            link_score_heatmap = image_utils.cvt2HeatmapImg(score_link)
            postprocess_time = time.time() - t0
            return {
                "boxes": boxes, "boxes_as_ratios": boxes_as_ratio, "polys": polys, "polys_as_ratios": polys_as_ratio,
                "heatmaps": {"text_score_heatmap": text_score_heatmap, "link_score_heatmap": link_score_heatmap},
                "times": {"resize_time": resize_time, "preprocessing_time": preprocessing_time, "craftnet_time": craftnet_time, "refinenet_time": refinenet_time, "postprocess_time": postprocess_time}
            }
        
        predict.get_prediction = exhaustive_patched_get_prediction
        
        # Also patch it in the root module since Craft class uses the root's reference
        import craft_text_detector
        craft_text_detector.get_prediction = exhaustive_patched_get_prediction
        
        print("[*] Applied compatibility patches for CRAFT and Torchvision.")
    except ImportError:
        pass
