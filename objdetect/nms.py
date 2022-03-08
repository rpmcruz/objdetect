def same(bi, bj):  # intersection-over-union
    intersection = (min(bi[2], bj[2])-max(bi[0], bj[0])) * (min(bi[3], bj[3])-max(bi[1], bj[1]))
    union = (bi[2]-bi[0])*(bi[3]-bi[1]) + (bj[2]-bj[0])*(bj[3]-bj[1]) - intersection
    return intersection / union

def NMS(B, H):
    lambda_nms = 0.5
    Bnms = []
    for bi, hi in zip(B, H):
        discard = False
        for bj, hj in zip(B, H):
            if same(bi, bj) > lambda_nms:
                if hj > hi:
                    discard = True
        if not discard:
            Bnms.append(bi)
    return Bnms
