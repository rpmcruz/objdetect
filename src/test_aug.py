import objdetect as od

dict_transform = od.aug.Compose([
    od.aug.Resize(int(256*1.05), int(256*1.05)),
    od.aug.RandomCrop(256, 256),
    od.aug.RandomHflip(),
    od.aug.RandomBrightnessContrast(0.1, 0.05),
])

tr = od.data.VOCDetection('/data', 'train', None, dict_transform)
d = tr[0]

od.plot.image(d['image'])
od.plot.bboxes(d['image'], d['bboxes'])
od.plot.classes(d['image'], d['bboxes'], d['classes'], tr.labels)
od.plot.show()
