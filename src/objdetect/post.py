def same(bi, bj):  # intersection-over-union
    intersection = (min(bi[2], bj[2])-max(bi[0], bj[0])) * (min(bi[3], bj[3])-max(bi[1], bj[1]))
    union = (bi[2]-bi[0])*(bi[3]-bi[1]) + (bj[2]-bj[0])*(bj[3]-bj[1]) - intersection
    return intersection / union

def NMS(C, B, *others, lambda_nms=0.5):
    Bnms = []
    Onms = [[] * len(others)]
    for i, (hi, bi) in enumerate(zip(C, B)):
        discard = False
        for j, (hj, bj) in enumerate(zip(C, B)):
            if same(bi, bj) > lambda_nms:
                if hj > hi:
                    discard = True
        if not discard:
            Bnms.append(bi)
            for oi, o in enumerate(others):
                Onms[oi].append(o[i])
    return Bnms, *Onms
