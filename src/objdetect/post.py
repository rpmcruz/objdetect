'''
Post-processing techniques to reduce the amount of superfluous bounding boxes.
'''

import torch

def valid_bboxes(bboxes):
    '''Converts the given bounding boxes into valid ones. That is, x2/y2 is swapped by x1/y1 if x2<x1 or y2<y1.'''
    return torch.stack((
        torch.clamp(torch.minimum(bboxes[:, 0], bboxes[:, 2]), min=0),
        torch.clamp(torch.minimum(bboxes[:, 1], bboxes[:, 3]), min=0),
        torch.clamp(torch.maximum(bboxes[:, 0], bboxes[:, 2]), max=1),
        torch.clamp(torch.maximum(bboxes[:, 1], bboxes[:, 3]), max=1),
    ), 1)

def same(bi, bj):
    '''Intersection over union utility.'''
    intersection = (min(bi[2], bj[2])-max(bi[0], bj[0])) * (min(bi[3], bj[3])-max(bi[1], bj[1]))
    union = (bi[2]-bi[0])*(bi[3]-bi[1]) + (bj[2]-bj[0])*(bj[3]-bj[1]) - intersection
    return intersection / union

def NMS(list_scores, list_bboxes, *list_others, lambda_nms=0.5):
    '''Non-Maximum Suppression (NMS) algorithm. It is a popular post-processing algorithm to clean-up similar bounding boxes.'''
    ret = [[] for _ in range(1+len(list_others))]
    for li, (scores, bboxes) in enumerate(zip(list_scores, list_bboxes)):
        ix = torch.tensor([
            not any(  # discard if all conditions met
                i != j and
                scores[j] > scores[i] and
                same(bboxes[i], bboxes[j]) >= lambda_nms
                for j in range(len(bboxes)))
            for i in range(len(bboxes))], dtype=bool)
        ret[0].append(bboxes[ix])
        for k, others in enumerate(list_others):
            ret[1+k].append(others[li][ix])
    return ret

if __name__ == '__main__':  # DEBUG
    import matplotlib.pyplot as plt
    import data, aug, plot
    ds = data.VOCDetection('/data', 'train', aug.Resize(256, 256))
    imgs = [ds[i]['image'] for i in range(4)]
    scores = [torch.rand(len(ds[i]['bboxes'])*3) for i in range(4)]
    bboxes = [[bbox+(torch.rand(4)-0.5)*0.05 for bbox in ds[i]['bboxes'] for _ in range(3)] for i in range(4)]
    classes = [[klass for klass in ds[i]['classes'] for _ in range(3)] for i in range(4)]
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plot.image(imgs[i])
        plot.bboxes(imgs[i], bboxes[i])
        plot.classes(imgs[i], bboxes[i], classes[i])
    plt.suptitle('Before NMS')
    plt.show()
    bboxes = [torch.stack(bb) for bb in bboxes]
    classes = [torch.tensor(kk) for kk in classes]
    bboxes, classes = NMS(scores, bboxes, classes)
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plot.image(imgs[i])
        plot.bboxes(imgs[i], bboxes[i])
        plot.classes(imgs[i], bboxes[i], classes[i])
    plt.suptitle('After NMS')
    plt.show()
