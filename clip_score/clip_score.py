import torch
import argparse
from tqdm import tqdm
from torchmetrics.multimodal.clip_score import CLIPScore
from common import prepare_imgpaths_prompts, load_images


def parse_args(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',         type=str,   default='./images',
                        help='images dir')
    parser.add_argument('--weights',        type=str,   default='',
                        help='initial weights path')
    parser.add_argument('--clip_model',     type=str,   default='openai/clip-vit-base-patch16',
                        help='clip model')
    parser.add_argument('--device',         type=str,   default='cuda',
                        help='device')


    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args



def main():
    args = parse_args()
    source, batch_size, clip_model, device = args.source, args.batch_size, args.clip_model, args.device

    ### Step1 prepare 
    imgpaths, prompts = prepare_imgpaths_prompts(source)
    score = 0
    metric = CLIPScore(model_name_or_path=clip_model).to(device)
    

    total_step = len(imgpaths) // batch_size + 1


    for i in tqdm(range(total_step)):

        score += metric(images_sub, prompts_sub).detach() * len(images_sub)

    print(f'clip score: {score / }')






if __name__ == '__main__':
    main()
