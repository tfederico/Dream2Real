import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Returns centre in (i, j) coordinates, rather than (x, y).
def centre_of_mass(binary_image):
    moments = cv2.moments(binary_image * 1.0)
    centre = np.array([int(moments["m01"] / moments["m00"]), int(moments["m10"] / moments["m00"])])
    return centre

# OPT: could be faster, since dilate, get_biggest_side and findCountours are done one pixel at a time.
# Maybe computing area and using that to determine when to stop would be faster.
def rescale_mask(mask, scale):
    if scale == 1.0:
        return mask

    mask = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    _, length = get_biggest_side(mask)
    new_length = length * scale
    if scale >= 1:
        while length < new_length:
            mask = cv2.dilate(mask, kernel, iterations=1)
            prev_length = length
            _, length = get_biggest_side(mask)
            if prev_length == length:
                return mask
            prev_length = length
    else:
        while length > new_length:
            mask = cv2.erode(mask, kernel, iterations=1)
            prev_length = length
            _, length = get_biggest_side(mask)
            if prev_length == length:
                return mask
            prev_length = length
    return mask

def get_biggest_side(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy().astype(np.uint8)
    mask_im = mask.copy() * 255
    contours, hierarchy = cv2.findContours(mask_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=len)
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int_(box)

    box_width = np.linalg.norm(box[0] - box[1])
    box_height = np.linalg.norm(box[1] - box[2])

    if box_width > box_height:
        return (box[2] - box[0]) / box_width, box_width
    else:
        return (box[3] - box[1]) / box_height, box_height

def get_smallest_side(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy().astype(np.uint8)
    mask_im = mask.copy() * 255
    contours, hierarchy = cv2.findContours(mask_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=len)
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int_(box)

    box_width = np.linalg.norm(box[0] - box[1])
    box_height = np.linalg.norm(box[1] - box[2])

    if box_width < box_height:
        return (box[2] - box[0]) / box_width, box_width
    else:
        return (box[3] - box[1]) / box_height, box_height

# From the Segment Anything documentation.
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# E.g. if bound is 0.5, then object bbox must not be outside the centre square of the image (of width 0.5 * img_width).
# Now batched!
def is_within_bounds_2d(poses, full_img_size, obj_img, bound):
    assert obj_img.shape[2] <= 4, f'Object image has shape {obj_img.shape}, but must have shape (H, W, C)'
    obj_width = (obj_img.shape[1] / full_img_size[1]) * 2
    obj_height = (obj_img.shape[0] / full_img_size[0]) * 2
    obj_centres = poses[:, :2] # pose may later contain orientation.
    obj_lefts = obj_centres[:, 0] - obj_width / 2
    obj_rights = obj_centres[:, 0] + obj_width / 2
    obj_tops = obj_centres[:, 1] + obj_height / 2
    obj_bottoms = obj_centres[:, 1] - obj_height / 2
    return (obj_lefts > (-1 + bound)).logical_and(obj_rights < (1 - bound)).logical_and(obj_tops < (1 - bound)).logical_and(obj_bottoms > (-1 + bound))

# Returns a crop with some background (for context) and no alpha channel. Used for captioning.
def get_thumbnail(img, obj_mask, padding=5, use_mask=True):
    if use_mask:
        img = img.clone()
        img[~obj_mask] = 255

    row_has_obj = torch.any(obj_mask.view(obj_mask.shape[0], -1), dim=-1)
    rows_with_obj = torch.where(row_has_obj)[0]
    first_obj_row = rows_with_obj[0]
    last_obj_row = rows_with_obj[-1]

    col_has_obj = torch.any(obj_mask.permute(1, 0).reshape(obj_mask.shape[1], -1), dim=-1)
    cols_with_obj = torch.where(col_has_obj)[0]
    first_obj_col = cols_with_obj[0]
    last_obj_col = cols_with_obj[-1]

    first_row = max(0, first_obj_row - padding)
    last_row = min(img.shape[0] - 1, last_obj_row + padding)
    first_col = max(0, first_obj_col - padding)
    last_col = min(img.shape[1] - 1, last_obj_col + padding)

    thumbnail = img[first_row:last_row + 1, first_col:last_col + 1]
    return thumbnail

# Post-processing for background mask due to seg association issues.
def remove_components_at_edges(mask):
    mask = mask.clone()

    num_comps, comp_img = cv2.connectedComponents(mask.cpu().numpy().astype(np.uint8))
    comp_img = torch.from_numpy(comp_img).to(mask.device)
    for i in range(num_comps):
        comp_mask = comp_img == i
        if mask_touches_edge(comp_mask):
            mask[comp_mask] = 0

    return mask

def mask_touches_edge(mask):
    row_has_obj = torch.any(mask.view(mask.shape[0], -1), dim=-1)
    rows_with_obj = torch.where(row_has_obj)[0]
    first_obj_row = rows_with_obj[0]
    last_obj_row = rows_with_obj[-1]

    col_has_obj = torch.any(mask.permute(1, 0).reshape(mask.shape[1], -1), dim=-1)
    cols_with_obj = torch.where(col_has_obj)[0]
    first_obj_col = cols_with_obj[0]
    last_obj_col = cols_with_obj[-1]

    return first_obj_row == 0 or last_obj_row == mask.shape[0] - 1 or first_obj_col == 0 or last_obj_col == mask.shape[1] - 1

