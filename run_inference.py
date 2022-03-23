import argparse
from path import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import models
from tqdm import tqdm

import torchvision.transforms as transforms
import flow_transforms
from imageio import imread, imwrite
import numpy as np
import cv2
from util import flow2rgb, create_pairs, concatenation

from skimage.measure import block_reduce

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch FlowNet inference on a folder of img pairs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR',
                    help='path to images folder, image names must match \'[name]0.[ext]\' and \'[name]1.[ext]\'')
parser.add_argument('pretrained', metavar='PTH', help='path to pre-trained model')
parser.add_argument('--output', '-o', metavar='DIR', default=None,
                    help='path to output folder. If not set, will be created in data folder')
parser.add_argument('--output-value', '-v', choices=['raw', 'vis', 'both'], default='both',
                    help='which value to output, between raw input (as a npy file) and color vizualisation (as an image file).'
                    ' If not set, will output both')
parser.add_argument('--div-flow', default=20, type=float,
                    help='value by which flow will be divided. overwritten if stored in pretrained file')
# parser.add_argument("--img-exts", metavar='EXT', default='bmp', nargs='*', type=str,
#                     help="images extensions to glob")
parser.add_argument('--img_exts', metavar='EXT', default='bmp', type=str,
                    help="images extensions to glob")
parser.add_argument('--max_flow', default=None, type=float,
                    help='max flow value. Flow map color is saturated above this value. If not set, will use flow map\'s max value')
parser.add_argument('--upsampling', '-u', choices=['nearest', 'bilinear', 'bicubic'], default='bicubic', help='if None, will output FlowNet raw input,'
                    'which is 4 times downsampled. If set, will output full resolution flow map, with selected upsampling')
parser.add_argument('--bidirectional', action='store_true', help='if set, will output invert flow (from 1 to 0) along with regular flow')
parser.add_argument('--seq_name', default='swan', type=str,
                    help='name of the sequence of interest')
parser.add_argument('--downsampling', default=None, type=int,
                    help='The downsampling factor for input image')
# parser.add_argument('--im_end', default=20, type=int,
#                     help='index of end image, the name of the image must be {name}-iii.img-ext (index of 3 numbers)')


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    global args, save_path
    args = parser.parse_args()

    if args.output_value == 'both':
        output_string = "raw output and RGB visualization"
    elif args.output_value == 'raw':
        output_string = "raw output"
    elif args.output_value == 'vis':
        output_string = "RGB visualization"
    print("=> will save " + output_string)
    data_dir = Path(args.data)
    print("=> fetching img pairs in '{}'".format(args.data))
    if args.output is None:
        save_path = data_dir/'flow'
    else:
        save_path = Path(args.output)
    print('=> will save everything to {}'.format(save_path))
    save_path.makedirs_p()
    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    img_pairs = create_pairs(data_dir, type_object = args.seq_name, ext = args.img_exts)
    # img_pairs = []
    # for ext in args.img_exts:
    #     test_files = data_dir.files('*1.{}'.format(ext))
    #     for file in test_files:
    #         img_pair = file.parent / (file.stem[:-1] + '2.{}'.format(ext))
    #         if img_pair.isfile():
    #             img_pairs.append([file, img_pair])
        

    print('{} pairs found'.format(len(img_pairs)))
    # create model
    network_data = torch.load(args.pretrained)
    print("=> using pre-trained model '{}'".format(network_data['arch']))
    model = models.__dict__[network_data['arch']](network_data).to(device)
    model.eval()
    cudnn.benchmark = True

    if 'div_flow' in network_data.keys():
        args.div_flow = network_data['div_flow']
        print("'{}' has default div_flow.".format(network_data['arch']))
    else:
        print("'{}' does not have default div_flow.".format(network_data['arch']))

    for (img1_file, img2_file) in tqdm(img_pairs):
        img1 = cv2.imread(img1_file)
        img2 = cv2.imread(img2_file)
        sigma_s = 60
        sigma_r = 0.3

        img1 = cv2.GaussianBlur(img1,(5,5),cv2.BORDER_DEFAULT)
        img2 = cv2.GaussianBlur(img2,(5,5),cv2.BORDER_DEFAULT)

        # img1 = cv2.edgePreservingFilter(img1, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)
        # img2 = cv2.edgePreservingFilter(img2, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)

        # img1 = cv2.detailEnhance(img1, sigma_s=sigma_s, sigma_r=sigma_r)
        # img2 = cv2.detailEnhance(img2, sigma_s=sigma_s, sigma_r=sigma_r)
        
        # img1 = cv2.pencilSketch(img1, sigma_s=sigma_s, sigma_r=sigma_r)
        # img2 = cv2.pencilSketch(img2, sigma_s=sigma_s, sigma_r=sigma_r)

        
        img1 = input_transform(img1)
        img2 = input_transform(img2)
        # print(img1.shape)
        if args.downsampling is not None:
            scale = args.downsampling
            img1 = F.avg_pool2d(img1, kernel_size=scale, stride=scale)
            img2 = F.avg_pool2d(img2, kernel_size=scale, stride=scale)
            # print(img1.shape)
            # img1 = input_transform(block_reduce(img1, block_size=(scale,scale,1), func=np.mean))
            # img2 = input_transform(block_reduce(img2, block_size=(scale,scale,1), func=np.mean))
            # img1 = input_transform(cv2.resize(img1,(int(img1.shape[1]/scale),int(img1.shape[0]/scale)), interpolation = cv2.INTER_AREA))
            # img2 = input_transform(cv2.resize(img2,(int(img2.shape[1]/scale),int(img2.shape[0]/scale)), interpolation = cv2.INTER_AREA))
            
        # else:
        #     img1 = input_transform(img1)
        #     img2 = input_transform(img2)
        # print(img1.size()[-2:], img1.size()[-2:]*4)
        # img1 = F.interpolate(img1, size=img1.size()[-2:]*4, mode=args.upsampling, align_corners=False)
        # img2 = F.interpolate(img2, size=img2.size()[-2:]*4, mode=args.upsampling, align_corners=False)
        # img1 = imread(img1_file)
        # img2 = imread(img2_file)
        # img1_4 = cv2.resize(img1,(np.shape(img1)[1]*4, np.shape(img1)[0]*4), cv2.INTER_CUBIC)
        # img2_4 = cv2.resize(img2,(np.shape(img2)[1]*4, np.shape(img2)[0]*4), cv2.INTER_CUBIC)
        # img1 = input_transform(img1_4)
        # img2 = input_transform(img2_4)
        input_var = torch.cat([img1, img2]).unsqueeze(0)

        if args.bidirectional:
            # feed inverted pair along with normal pair
            inverted_input_var = torch.cat([img2, img1]).unsqueeze(0)
            input_var = torch.cat([input_var, inverted_input_var])
            # input_var = torch.cat([img2, img1]).unsqueeze(0)

        input_var = input_var.to(device)
        # compute output
        output = model(input_var)
        # if args.downsampling is not None:
        #     scale = args.downsampling
        #     output /= scale
        if args.upsampling is not None:
            output = F.interpolate(output, size=img1.size()[-2:], mode=args.upsampling, align_corners=False)
        for suffix, flow_output in zip(['flow', 'inv_flow'], output):
            # filename = save_path/'{}{}'.format(img1_file.stem[:-1], suffix)
            filename = save_path/'{}{}'.format(Path(img2_file).stem, suffix)
            if args.output_value in['vis', 'both']:
                rgb_flow = flow2rgb(args.div_flow * flow_output, max_value=args.max_flow)
                to_save = (rgb_flow * 255).astype(np.uint8).transpose(1,2,0)
                imwrite(filename + '.png', to_save)
            if args.output_value in ['raw', 'both']:
                # Make the flow map a HxWx2 array as in .flo files
                to_save = (args.div_flow*flow_output).cpu().numpy().transpose(1,2,0)
                np.save(filename + '.npy', to_save)


if __name__ == '__main__':
    main()
