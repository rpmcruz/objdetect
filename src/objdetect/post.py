'''
Post-processing techniques to reduce the amount of superfluous bounding boxes.
'''

def same(bi, bj):
    '''Intersection over union utility.'''
    intersection = (min(bi[2], bj[2])-max(bi[0], bj[0])) * (min(bi[3], bj[3])-max(bi[1], bj[1]))
    union = (bi[2]-bi[0])*(bi[3]-bi[1]) + (bj[2]-bj[0])*(bj[3]-bj[1]) - intersection
    return intersection / union

def NMS(H, B, *others, lambda_nms=0.5):
    '''Non-Maximum Suppression (NMS) algorithm. It is a popular post-processing algorithm to clean-up similar bounding boxes. The `others` parameter allows passing other values which are also returned together with the filtered bounding boxes.'''
    Hnms = []
    Bnms = []
    Onms = [[] * len(others)]
    for i, (hi, bi) in enumerate(zip(H, B)):
        discard = False
        for j, (hj, bj) in enumerate(zip(H, B)):
            if i != j and same(bi, bj) >= lambda_nms and hj > hi:
                discard = True
                break
        if not discard:
            Hnms.append(hi)
            Bnms.append(bi)
            for oi, o in enumerate(others):
                Onms[oi].append(o[i])
    return Hnms, Bnms, *Onms
