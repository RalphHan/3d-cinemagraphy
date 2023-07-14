import gradio as gr
import argparse
import os.path
import imageio
import torch.utils.data.distributed
from tqdm import tqdm
import cv2
from predictor import MaskFormer,OpticalPridictor, DPTModel, Ruler, SAM

from lib.config import load_config
from lib.utils.general_utils import check_file
from lib.model.inpaint.model import SpaceTimeAnimationModel
from lib.model.motion.motion_model import SPADEUnetMaskMotion
from lib.model.motion.sync_batchnorm import convert_model
from lib.renderer import ImgRenderer
from lib.model.inpaint.inpainter import Inpainter
from lib.utils.data_utils import resize_img
import numpy as np
import uuid
from PIL import Image

class Animator:
    def __init__(self,args):
        self.args= args
        check_file(args.config)
        config = load_config(args.config)
        self.mf=MaskFormer()
        self.sam=SAM()
        self.op=OpticalPridictor()
        self.dpt=DPTModel()
        self.ruler=Ruler(config["data"])
        model = SpaceTimeAnimationModel(args, config)
        scene_flow_estimator = SPADEUnetMaskMotion(config['generator']).to('cuda')
        scene_flow_estimator = convert_model(scene_flow_estimator)
        scene_flow_estimator_weight = torch.load('ckpts/sceneflow_model.pth',
                                                 map_location='cuda')
        scene_flow_estimator.load_state_dict(scene_flow_estimator_weight['netG'])
        inpainter = Inpainter(device='cuda')
        self.renderer = ImgRenderer(args, config, model, scene_flow_estimator, inpainter, 'cuda')
        model.switch_to_eval()

    def get_input_data(self,prompts, image,video_out_folder):
        motion_rgb = self.ruler.motion_input_transform(Image.fromarray(image))
        mask = torch.zeros((768, 768), dtype=torch.bool)
        for prompt in prompts.strip().split(','):
            mask |= self.mf.inference(image,prompt.strip())[0,0].bool()
        mask = self.sam.inference(image,mask)
        hints = self.op.get_hint(image, mask)
        src_depth = self.dpt.run_dpt(image,video_out_folder)
        src_img = image / 255.
        src_img = resize_img(src_img, 1)
        h, w = src_img.shape[:2]
        intrinsic = np.array([[max(h, w), 0, w // 2],
                              [0, max(h, w), h // 2],
                              [0, 0, 1]])
        pose = np.eye(4)
        return {
            'motion_rgbs': motion_rgb[None, ...],
            'src_img': self.ruler.to_tensor(src_img).float()[None],
            'src_depth': self.ruler.to_tensor(src_depth).float()[None],
            'hints': hints[0],
            'mask': mask[0],
            'intrinsic': torch.from_numpy(intrinsic).float()[None],
            'pose': torch.from_numpy(pose).float()[None],
            'scale_shift': torch.tensor([1., 0.]).float()[None],
        }


    def render(self, prompts, image):
        fh, fw = image.shape[:2]
        if image.shape[1] > 768:
            fh, fw = int(768 * fh / fw), 768
            image = cv2.resize(image, (fw, fh))
        device='cuda'
        the_uuid=str(uuid.uuid4())
        # set up folder
        video_out_folder = os.path.join('demo', 'gradio_videos', the_uuid)
        os.makedirs(video_out_folder, exist_ok=True)
        data = self.get_input_data(prompts, image, video_out_folder)
        torch.cuda.empty_cache()
        with torch.no_grad():
            self.renderer.process_data(data)
            coord, flow, pts_src, featmaps_src, rgba_layers_src, depth_layers_src, mask_layers_src = \
                self.renderer.compute_flow_and_inpaint()
            flow = flow / args.flow_scale

            num_frames = [60]
            video_paths = ['still']
            Ts = [
                np.zeros((60,3),dtype=np.float64)
            ]
            crop = 5
            kernel = torch.ones(5, 5, device=device)

            for j, T in enumerate(Ts):
                T = torch.from_numpy(T).float().to(self.renderer.device)
                time_steps = range(0, num_frames[j])
                start_index = torch.tensor([0]).to(device)
                end_index = torch.tensor([num_frames[j] - 1]).to(device)
                frames = []
                for middle_index, t_step in tqdm(enumerate(time_steps), total=len(time_steps), ncols=150,
                                                 desc='generating video of {} camera trajectory'.format(video_paths[j])):
                    middle_index = torch.tensor([middle_index]).to(device)
                    time = ((middle_index.float() - start_index.float()).float() / (
                            end_index.float() - start_index.float() + 1.0).float()).item()

                    flow_f = self.renderer.euler_integration(flow, middle_index.long() - start_index.long())
                    flow_b = self.renderer.euler_integration(-flow, end_index.long() + 1 - middle_index.long())
                    flow_f = flow_f.permute(0, 2, 3, 1)
                    flow_b = flow_b.permute(0, 2, 3, 1)

                    _, all_pts_f, _, all_rgbas_f, _, all_feats_f, \
                        all_masks_f, all_optical_flow_f = \
                        self.renderer.compute_scene_flow_for_motion(coord, torch.inverse(self.renderer.pose), self.renderer.src_img,
                                                               rgba_layers_src, featmaps_src, pts_src, depth_layers_src,
                                                               mask_layers_src, flow_f, kernel, with_inpainted=True)
                    _, all_pts_b, _, all_rgbas_b, _, all_feats_b, \
                        all_masks_b, all_optical_flow_b = \
                        self.renderer.compute_scene_flow_for_motion(coord, torch.inverse(self.renderer.pose), self.renderer.src_img,
                                                               rgba_layers_src, featmaps_src, pts_src, depth_layers_src,
                                                               mask_layers_src, flow_b, kernel, with_inpainted=True)

                    all_pts_flowed = torch.cat(all_pts_f + all_pts_b)
                    all_rgbas_flowed = torch.cat(all_rgbas_f + all_rgbas_b)
                    all_feats_flowed = torch.cat(all_feats_f + all_feats_b)
                    all_masks = torch.cat(all_masks_f + all_masks_b)
                    all_side_ids = torch.zeros_like(all_masks.squeeze(), dtype=torch.long)
                    num_pts_2 = sum([len(x) for x in all_pts_b])
                    all_side_ids[-num_pts_2:] = 1

                    pred_img, _, meta = self.renderer.render_pcd(all_pts_flowed,
                                                            all_rgbas_flowed,
                                                            all_feats_flowed,
                                                            all_masks, all_side_ids,
                                                            t=T[middle_index.item()],
                                                            time=time,
                                                            t_step=t_step,
                                                            path_type=video_paths[j])

                    frame = (255. * pred_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()).astype(np.uint8)
                    frame = frame[crop:-crop, crop:-crop]
                    frames.append(frame)

                video_out_file = os.path.join(video_out_folder,
                                              f'video.mp4')

                imageio.mimwrite(video_out_file, frames, fps=25, quality=8)
                return video_out_file,cv2.resize(data['mask'][0].numpy(),(fw,fh))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ########## general ##########
    parser.add_argument('-c', '--config', type=str, default='configs/config.yaml', help='config file path')
    parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--distributed', action='store_true', help='if use distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='rank for distributed training')

    parser.add_argument('--save_frames', action='store_true', help='if save frames')
    parser.add_argument('--correct_inpaint_depth', action='store_true',
                        help='use this option to correct the depth of inpainting area to prevent occlusion')
    parser.add_argument("--flow_scale", type=float, default=1.0,
                        help='flow scale that used to control the speed of fluid')
    parser.add_argument("--ds_factor", type=float, default=1.0,
                        help='downsample factor for the input images')

    ########## checkpoints ##########
    parser.add_argument("--ckpt_path", type=str, default='ckpts/model_150000.pth',
                        help='specific weights file to reload')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_load_opt", action='store_true',
                        help='do not load optimizer when reloading')
    parser.add_argument("--no_load_scheduler", action='store_true',
                        help='do not load scheduler when reloading')
    args = parser.parse_args()
    am=Animator(args)
    demo = gr.Interface(am.render, [gr.Textbox("river,sky"),
                                    gr.Image(value=cv2.cvtColor(cv2.imread("./demo/0/image.png"), cv2.COLOR_BGR2RGB))],
                        [gr.Video(format="mp4", autoplay=True),gr.Image()])
    demo.launch(server_name='0.0.0.0', server_port=7864)
