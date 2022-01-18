#testing yolo in/output
from pytorchyolo import detect, models
import torch
from utils import *
import torchsummary
# Load the YOLO model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo = models.load_model(
  "./yolov3-custom.cfg")
# print(yolo)
seq_len, batch_size,num_box = 1,1,1
img_size = 512
#--------training---------
test_img = torch.rand(seq_len*batch_size,1,512,512)
out = yolo(test_img)
# print(out[0].shape) #1,3,16,16,6
# print(out[1].shape) #1,3,32,32,6
# print(out[2].shape) #1,3,64,64,6
# print(out[0][0,1,0,0,:])
# 6 = 1(# classes)+4(bboxes)+1(object score)
# 16/32/64 grid cell size at differet scales
# a sample index is added when feeding into yolo
bbox = torch.rand(seq_len*batch_size,4) 
image_label = torch.arange(seq_len).reshape(-1,1)
label = torch.ones(seq_len*batch_size,1)
target = torch.concat([image_label, label, bbox],dim=1)
target.to(device)
loss, loss_components = compute_loss(out, target, yolo)
print(loss)
loss.backward()
#----------testing-----------
sample_metrics = []  # List of tuples (TP, confs, pred)
labels = []
#extract class labels
labels += target[:, 1].tolist()
#scale target to original image size
target[:, 2:] = xywh2xyxy(target[:, 2:])
target[:, 2:] *= img_size
yolo.eval() #yolo handle concatenating boxes in eval mode
out = yolo(test_img)
conf_thres, iou_thres = 0.25, 0.45
print(out.shape, target.shape)
out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres)
sample_metrics += get_batch_statistics(out, target, iou_threshold=iou_thres)
true_positives, pred_scores, pred_labels = [
    np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
print(true_positives.shape)
print(true_positives)
metrics_output = ap_per_class(
    true_positives, pred_scores, pred_labels, labels)

precision, recall, AP, f1, ap_class = metrics_output
print(precision, recall, AP, f1, ap_class)
torchsummary.summary(yolo, input_size=(1,512,512))