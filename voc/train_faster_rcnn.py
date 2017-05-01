import matplotlib
matplotlib.use('agg')
import fire
import numpy as np

import chainer
from chainer import training
from chainer.training import extensions
from chainer.training import updaters
from chainer.training.triggers.manual_schedule_trigger import ManualScheduleTrigger

from chainercv.datasets import TransformDataset
from chainercv.datasets import VOCDetectionDataset
from chainercv.extensions import DetectionVisReport
from chainercv import transforms
from chainercv.datasets.pascal_voc.voc_utils import pascal_voc_labels

from chainercv.links import FasterRCNNResNet

from extensions.detection_report import DetectionReport
from faster_rcnn_deformable import FasterRCNNResNetDeformable
from multiple_dataset import MultipleDataset


class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


mean_pixel = np.array([102.9801, 115.9465, 122.7717])[:, None, None]


def get_train_iter(device, train_data, batchsize=1, loaderjob=None):
    if len(device) > 1:
        train_iter = [
            chainer.iterators.MultiprocessIterator(
                i, batchsize, n_processes=loaderjob, shared_mem=10000000)
            for i in chainer.datasets.split_dataset_n_random(train_data, len(device))]
    else:
        train_iter = chainer.iterators.MultiprocessIterator(
            train_data, batch_size=batchsize, n_processes=loaderjob, shared_mem=100000000)
    return train_iter


def get_updater(train_iter, optimizer, device):
    if len(device) > 1:
        updater = updaters.MultiprocessParallelUpdater(
            train_iter, optimizer, devices=device)
    else:
        updater = chainer.training.updater.StandardUpdater(
            train_iter, optimizer, device=device[0])
    return updater


def main(gpus=[0, 1, 2], model_mode='resnet',
         lr=1e-3, gamma=0.1,
        out='result', resume='', long_mode=False):
    for key, val in locals().items():
        print('{}: {}'.format(key, val))
    batch_size = 1
    if not isinstance(gpus, list or tuple):
        gpus = [gpus]
    lr = lr / float(len(gpus))  # adjust lr

    iteration = 240000 / len(gpus)  # 30k * batch_size=8
    step_size = int(iteration * (2. / 3.))
    labels = pascal_voc_labels
    train_07_data = VOCDetectionDataset(mode='trainval', year='2007')
    train_12_data = VOCDetectionDataset(mode='trainval', year='2012')
    train_data = MultipleDataset([train_07_data, train_12_data])
    test_data = VOCDetectionDataset(
        mode='test', year='2007',
        use_difficult=True, return_difficult=True)

    def get_transform(use_random):
        def transform(in_data):
            has_difficult = len(in_data) == 4
            img, bbox, label = in_data[:3]
            img -= mean_pixel 
            # Resize bounding box to a shape
            # with the smaller edge at least at length 600
            _, H, W = img.shape
            img = transforms.scale(img, 600)
            _, o_H, o_W = img.shape
            # Prevent the biggest axis from being more than MAX_SIZE
            if max(o_H, o_W) > 1000:
                rate = 1000 / float(max(o_H, o_W))
                img = transforms.resize(img, (int(o_W * rate), int(o_H * rate)))
                _, o_H, o_W = img.shape
            bbox = transforms.resize_bbox(bbox, (W, H), (o_W, o_H))

            # horizontally flip
            if use_random:
                img, params = transforms.random_flip(img, x_random=True, return_param=True)
                bbox = transforms.flip_bbox(bbox, (o_W, o_H), params['x_flip'])

            scale = float(o_W) / float(W)

            if has_difficult:
                return img, bbox, label, scale, in_data[-1]
            return img, bbox, label, scale
        return transform

    train_data = TransformDataset(train_data, get_transform(True))
    test_data = TransformDataset(test_data, get_transform(False))

    if model_mode == 'vgg':
        model = FasterRCNNVGG(n_class=len(labels))
        weight_decay = 0.0005
    elif model_mode == 'resnet':
        model = FasterRCNNResNet(n_class=len(labels))
        weight_decay = 0.0001
    elif model_mode == 'deformable':
        model = FasterRCNNResNetDeformable(n_class=len(labels))
        weight_decay = 0.0001

    if len(gpus) == 1:
        model.to_gpu(gpus[0])
        chainer.cuda.get_device(gpus[0]).use()
    optimizer = chainer.optimizers.MomentumSGD(lr=lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=weight_decay))

    train_iter = get_train_iter(gpus, train_data, batch_size)
    test_iter = chainer.iterators.MultiprocessIterator(
        test_data, batch_size=1, repeat=False, shared_mem=10000000)
    updater = get_updater(train_iter, optimizer, gpus)

    trainer = training.Trainer(updater, (iteration, 'iteration'), out=out)

    # only save the lastest model
    import datetime
    fn = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '_')
    trainer.extend(extensions.snapshot(filename='snapshot'+fn), trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model, 'model' + fn), trigger=(1, 'epoch'))
    trainer.extend(extensions.ExponentialShift('lr', gamma),
                   trigger=(step_size, 'iteration'))

    log_interval = 20, 'iteration'
    val_interval = 70000, 'iteration'
    plot_interval = 3000, 'iteration'
    print_interval = 20, 'iteration'

    trainer.extend(
        chainer.training.extensions.observe_lr(),
        trigger=log_interval
    )
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss',
         'main/loss_bbox',
         'main/loss_cls',
         'main/rpn_loss_cls',
         'main/rpn_loss_bbox',
         'map'
         ]), trigger=print_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # visualize training
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss'],
                file_name='loss.png', trigger=plot_interval
            ),
            trigger=plot_interval
        )
        trainer.extend(
            extensions.PlotReport(
                ['main/rpn_loss_cls'],
                file_name='rpn_loss_cls.png', trigger=plot_interval
            ),
            trigger=plot_interval
        )
        trainer.extend(
            extensions.PlotReport(
                ['main/rpn_loss_bbox'],
                file_name='rpn_loss_bbox.png'
            ),
            trigger=plot_interval
        )
        trainer.extend(
            extensions.PlotReport(
                ['main/loss_cls'],
                file_name='loss_cls.png'
            ),
            trigger=plot_interval
        )
        trainer.extend(
            extensions.PlotReport(
                ['main/loss_bbox'],
                file_name='loss_bbox.png'
            ),
            trigger=plot_interval
        )
        trainer.extend(
            extensions.PlotReport(
                ['main/map'],
                file_name='map.png'
            ),
            trigger=val_interval
        )

    def post_transform(in_data):
        img, bbox, label, scale, difficult = in_data
        _, H, W = img.shape
        o_W = int(W / scale)
        o_H = int(H / scale)
        bbox = transforms.resize_bbox(bbox, (W, H), (o_W, o_H))
        return img, bbox, label, scale, difficult
        
    use_07_metric = True
    trainer.extend(
        DetectionReport(
            model, test_data,gpus[0], len(labels), minoverlap=0.5,
            use_07_metric=use_07_metric, post_transform=post_transform),
        trigger=val_interval, invoke_before_training=False

    )

    trainer.extend(extensions.dump_graph('main/loss'))

    if resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(resume, trainer)

    trainer.run()


if __name__ == '__main__':
    fire.Fire(main)
