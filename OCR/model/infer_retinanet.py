#!/usr/bin/env python
# coding: utf-8
try:
    import fitz
except:
    pass
import sys
import local_config
sys.path.append(local_config.global_3rd_party)
from os.path import join
from ovotools.params import AttrDict
import numpy as np
from collections import OrderedDict
import torch
import timeit
import copy
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import PIL.ImageOps
import data_utils.data as data
import braille_utils.label_tools as lt
import model.create_model_retinanet as create_model_retinanet
import pytorch_retinanet as pytorch_retinanet
import pytorch_retinanet.encoder
import braille_utils.postprocess as postprocess

VALID_IMAGE_EXTENTIONS = tuple('.jpg,.jpe,.jpeg,.png,.gif,.svg,.bmp,.tiff,.tif,.jfif'.split(','))
inference_width = 1024
model_weights = 'model.t7'
params_fn = join(local_config.data_path, 'weights', 'param.txt')
model_weights_fn = join(local_config.data_path, 'weights', model_weights)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cls_thresh = 0.3
nms_thresh = 0.02
REFINE_COEFFS = [0.083, 0.092, -0.083, -0.013]
class BraileInferenceImpl(torch.nn.Module):
    def __init__(self, params, model, device, label_is_valid, verbose=0):
        super(BraileInferenceImpl, self).__init__()
        self.verbose = verbose
        self.device = device
        if isinstance(model, torch.nn.Module):
            self.model_weights_fn = ""
            self.model = model
        else:
            self.model_weights_fn = model
            self.model, _, _ = create_model_retinanet.create_model_retinanet(params, device=device)
            self.model = self.model.to(device)
            self.model.load_state_dict(torch.load(self.model_weights_fn, map_location='cpu'))
        self.model.eval()

        self.encoder = pytorch_retinanet.encoder.DataEncoder(**params.model_params.encoder_params)
        self.valid_mask = torch.tensor(label_is_valid).long()
        self.cls_thresh = cls_thresh
        self.nms_thresh = nms_thresh
        self.num_classes = [] if not params.data.get('class_as_6pt', False) else [1]*6

    def forward(self, input_tensor, find_orientation, process_2_sides):
        t = timeit.default_timer()
        input_data = input_tensor.unsqueeze(0)

        loc_pred, cls_pred = self.model(input_data)
        
        best_idx = 0
        err_score = (torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.]))

        h, w = input_data.shape[2:]
        boxes, labels, scores = self.encoder.decode(loc_pred[0].cpu().data,
                                                    cls_pred[0].cpu().data, (w, h),
                                                    cls_thresh=self.cls_thresh, nms_thresh=self.nms_thresh,
                                                    num_classes=self.num_classes)
        if len(self.num_classes) > 1:
            labels = torch.tensor([lt.label010_to_int([str(s.item()+1) for s in lbl101]) for lbl101 in labels])
        if process_2_sides:
            boxes2, labels2, scores2 = self.encoder.decode(loc_pred[0].cpu().data,
                                                           cls_pred[0].cpu().data, (w, h),
                                                           cls_thresh=self.cls_thresh, nms_thresh=self.nms_thresh,
                                                           num_classes=self.num_classes)
        else:
            boxes2, labels2, scores2 = None, None, None
        if self.verbose >= 2:
            print("        forward.decode", timeit.default_timer() - t)
            t = timeit.default_timer()
        return boxes, labels, scores, best_idx, err_score, boxes2, labels2, scores2


class BrailleInference:

    DRAW_NONE = 0
    DRAW_ORIGINAL = 1
    DRAW_REFINED = 2
    DRAW_BOTH = DRAW_ORIGINAL | DRAW_REFINED
    DRAW_FULL_CHARS = 4

    def __init__(self, params_fn=params_fn, model_weights_fn=model_weights_fn, create_script=None,
                 verbose=1, inference_width=inference_width, device=device):
        self.verbose = verbose
        if not torch.cuda.is_available() and device != 'cpu':
            print('CUDA not available. CPU is used')
            device = 'cpu'

        params = AttrDict.load(params_fn, verbose=verbose)
        params.data.net_hw = (inference_width, inference_width,)
        params.data.batch_size = 1
        params.augmentation = AttrDict(
            img_width_range=(inference_width, inference_width),
            stretch_limit=0.0,
            rotate_limit=0,
        )
        self.preprocessor = data.ImagePreprocessor(params, mode='train')

        if isinstance(model_weights_fn, torch.nn.Module):
            self.impl = BraileInferenceImpl(params, model_weights_fn, device, lt.label_is_valid, verbose=verbose)
        else:
            model_script_fn = model_weights_fn + '.pth'
            if create_script != False:
                self.impl = BraileInferenceImpl(params, model_weights_fn, device, lt.label_is_valid, verbose=verbose)
                if create_script is not None:
                    self.impl = torch.jit.script(self.impl)
                if isinstance(self.impl, torch.jit.ScriptModule):
                    torch.jit.save(self.impl, model_script_fn)
                    if verbose >= 1:
                        print("Model loaded and saved to " + model_script_fn)
                else:
                    if verbose >= 1:
                        print("Model loaded")
            else:
                self.impl = torch.jit.load(model_script_fn)
                if verbose >= 1:
                    print("Model pth loaded")
        self.impl.to(device)
        self.result = None
        
    def get_result(self):
        return self.result

    def load_pdf(self, img_fn):
        try:
            pdf_file = fitz.open(img_fn)
            pg = pdf_file.loadPage(0)
            pdf = pg.getPixmap()
            cspace = pdf.colorspace
            if cspace is None:
                mode = "L"
            elif cspace.n == 1:
                mode = "L" if pdf.alpha == 0 else "LA"
            elif cspace.n == 3:
                mode = "RGB" if pdf.alpha == 0 else "RGBA"
            else:
                mode = "CMYK"
            img = PIL.Image.frombytes(mode, (pdf.width, pdf.height), pdf.samples)
            return img
        except Exception as exc:
            return None


    def run(self, img, lang, draw_refined, find_orientation, process_2_sides, align_results, repeat_on_aligned=True, gt_rects=[]):
        results_dict = self.run_impl(img, lang, draw_refined, find_orientation,
                                        process_2_sides=process_2_sides, align=align_results, draw=True, gt_rects=gt_rects)
        return results_dict

    	
    def refine_lines(self, lines):
        for ln in lines:
            for ch in ln.chars:
                h = ch.refined_box[3] - ch.refined_box[1]
                coefs = np.array(REFINE_COEFFS)
                deltas = h * coefs
                ch.refined_box = (np.array(ch.refined_box) + deltas).tolist()

    def run_impl(self, img, lang, draw_refined, find_orientation, process_2_sides, align, draw, gt_rects=[]):
        t = timeit.default_timer()
        np_img = np.asarray(img)
        
        if (len(np_img.shape) > 2 and np_img.shape[2] < 3):  # grayscale -> reduce dim
            np_img = np_img[:,:,0]
        aug_img, aug_gt_rects = self.preprocessor.preprocess_and_augment(np_img, gt_rects)
        aug_img = data.unify_shape(aug_img)
        input_tensor = self.preprocessor.to_normalized_tensor(aug_img, device=self.impl.device)
        
        with torch.no_grad():
            boxes, labels, scores, best_idx, err_score, boxes2, labels2, scores2 = self.impl(
                input_tensor, find_orientation=False, process_2_sides=process_2_sides)
        
        boxes = boxes.tolist()
        labels = labels.tolist()
        scores = scores.tolist()
        lines = postprocess.boxes_to_lines(boxes, labels, lang=lang)
        self.refine_lines(lines)

        aug_img = PIL.Image.fromarray(aug_img)
        
        if align and not process_2_sides:
            hom = postprocess.find_transformation(lines, (aug_img.width, aug_img.height))
            if hom is not None:
                aug_img = postprocess.transform_image(aug_img, hom)
                boxes = postprocess.transform_rects(boxes, hom)
                lines = postprocess.boxes_to_lines(boxes, labels, lang=lang)
                self.refine_lines(lines)
                aug_gt_rects = postprocess.transform_rects(aug_gt_rects, hom)
        else:
            hom = None
                    
        results_dict = {
            'image': aug_img,
            'best_idx': 0,  # Always NONE
            'err_scores': list([ten.cpu().data.tolist() for ten in err_score]),
            'gt_rects': aug_gt_rects,
            'homography': hom.tolist() if hom is not None else hom,
        }

        results_dict.update(self.draw_results(aug_img, boxes, lines, labels, scores, False, draw_refined))
        
        return results_dict


    def draw_results(self, aug_img, boxes, lines, labels, scores, reverse_page, draw_refined):
        suff = '.rev' if reverse_page else ''
        aug_img = copy.deepcopy(aug_img)
        draw = PIL.ImageDraw.Draw(aug_img)
        for ln in lines:
            for ch in ln.chars:
                draw.rectangle(list(ch.refined_box), outline='green')
        return {
            'labeled_image' + suff: aug_img,
            'lines' + suff: lines,
            'text' + suff: "",
            'braille' + suff: [],
            'dict' + suff: self.to_dict(aug_img, lines, draw_refined),
            'boxes' + suff: boxes,
            'labels' + suff: labels,
            'scores' + suff: scores,
        }


    def to_dict(self, img, lines, draw_refined = DRAW_NONE):
        shapes = []
        for ln in lines:
            for ch in ln.chars:
                ch_box = ch.refined_box if (draw_refined & self.DRAW_BOTH) != self.DRAW_ORIGINAL else ch.original_box
                shape = {
                    "label": ch.label,
                    "points": [[ch_box[0], ch_box[1]],
                               [ch_box[2], ch_box[3]]],
                    "shape_type": "rectangle",
                    "line_color": None,
                    "fill_color": None,
                }
                shapes.append(shape)
        res = {"shapes": shapes,
               "imageHeight": img.height, "imageWidth": img.width, "imagePath": None, "imageData": None,
               "lineColor": None, "fillColor": None,
               }
        return res
