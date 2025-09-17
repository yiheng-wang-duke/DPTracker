from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
import torch.nn.functional as F
import torch.nn as nn
from lib.models.dptrack_t.loss_functions import DJSLoss
from lib.models.dptrack_t.statistics_network import (
    GlobalStatisticsNetwork,
)


class DPTrackTActor(BaseActor):
    """ Actor for training DPTrackT models """
    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        # print(loss_weight)
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.djs_loss = DJSLoss()
        self.feature_map_size = 16  #128x128
        self.feature_map_channels = 240
        if hasattr(self.net, 'module'):
            # support DDP
            self.num_ch_coding = self.net.module.backbone.embed_dim
        else:
            self.num_ch_coding = self.net.backbone.embed_dim
        self.coding_size = 8
        self.global_stat_x = GlobalStatisticsNetwork(
            feature_map_size=self.feature_map_size,
            feature_map_channels=self.feature_map_channels,
            coding_channels=self.num_ch_coding,
            coding_size=self.coding_size,
        ).cuda()
        self.s = 'maxmean'

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # print('data.keys()', data.keys())
        # return
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        template_eva_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_eva_img_i = data['template_eva_images'][i].view(-1,
                                                             *data['template_eva_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)
            template_eva_list.append(template_eva_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_eva_img = data['search_eva_images'][0].view(-1, *data['search_eva_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        template_anno=data['template_anno']
        template_eva_anno=data['template_eva_anno']
        search_anno=data['search_anno']
        search_eva_anno=data['search_eva_anno']


        if len(template_list) == 1:
            template_list = template_list[0]
            template_eva_list = template_eva_list[0]

        out_dict = self.net(template=template_list,
                            search=search_img, 
                            template_anno=template_anno, 
                            search_anno=search_anno, 
                            is_distill=False,
                            training_dataset=data['dataset'])
        return out_dict

    def compute_adw_loss(self, ious, training_datasets):
        trainging_samples = {'exdark':0.74e4, 'bdd100k_night':2.81e4, 'shift_night':1.53e4, 'got10k':106e4, 'lasot':224e4, 'trackingnet':1410e4, 'coco':11.8e4, 'visdrone':7e4}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        N_max = torch.tensor(max(trainging_samples.values())).to(device)

        adw_loss = []
        for idx, dataset in enumerate(training_datasets):
            iou = ious[idx] if ious[idx] != 0 else ious[idx] + 0.01
            samples_number = torch.tensor(trainging_samples.get(dataset)).to(device)
            loss = torch.pow(torch.log(N_max) / torch.log(samples_number) + 0.5, 1 - iou) * torch.log(iou) - iou * (1 - iou)
            adw_loss.append(loss)
        return -torch.stack(adw_loss).mean()

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        mine_loss = pred_dict['mine_loss']
        activeness_loss = pred_dict['activeness_loss']
        #######################
        loss = self.loss_weight['giou'] * giou_loss \
               + self.loss_weight['l1'] * l1_loss \
               + self.loss_weight['focal'] * location_loss \
               + 0.0001 * mine_loss \
               + 50 * activeness_loss 

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
