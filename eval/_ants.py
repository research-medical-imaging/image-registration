import ants
import numpy as np

def ants_pred(fixed, moving, seg2):
    """return warped and w_seg2 """
    warps   = []
    w_seg2s = []
    flows   = []
    
    for i in range(fixed.shape[0]):
        im_fixed  = ants.from_numpy(fixed.cpu().numpy()[i,0])
        im_moving = ants.from_numpy(moving.cpu().numpy()[i,0])
        reg       = ants.registration(fixed=im_fixed, moving=im_moving, type_of_transform='SyN')
        warped    = ants.apply_transforms(fixed=im_fixed, moving=im_moving, transformlist=reg['fwdtransforms'])
        warps.append(warped.numpy())
        w_seg2    = ants.apply_transforms(fixed=im_fixed, moving=ants.from_numpy(seg2.cpu().numpy()[i,0]), transformlist=reg['fwdtransforms'])
        w_seg2s.append(w_seg2.numpy())
        flow      = ants.image_read(reg['fwdtransforms'][0])
        flows.append(flow.numpy())
    
    w_seg2 = np.array(w_seg2s)[:, None]                         # 1 x 1 x 192 x 192 x 208
    warped = np.array(warps)[:, None]                           # 1 x 1 x 192 x 192 x 208
    flow   = np.transpose(np.array(flows), (0, 4, 1, 2, 3))     # 1 x 3 x 192 x 192 x 208
    
    return {
        "warped": warped,
        "w_seg2": w_seg2,
        "flow":   flow,
    }