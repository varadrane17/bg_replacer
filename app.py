import gradio as gr
import torch
from diffusers import (
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
)
from diffusers.utils import load_image
from replace_bg.model.pipeline_controlnet_sd_xl import StableDiffusionXLControlNetPipeline
from replace_bg.model.controlnet import ControlNetModel
from replace_bg.utilities import resize_image, remove_bg_from_image, paste_fg_over_image, get_control_image_tensor
from huggingface_hub import login
import os 


api_token = "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"
login(api_token)


controlnet = ControlNetModel.from_pretrained("briaai/BRIA-2.3-ControlNet-BG-Gen", torch_dtype=torch.float16)  #"briaai/BRIA-2.3
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained("SG161222/RealVisXL_V4.0_Lightning", controlnet=controlnet, torch_dtype=torch.float16, vae=vae).to('cuda:0')
pipe.scheduler = EulerAncestralDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    steps_offset=1
)

def generate_(prompt, negative_prompt, control_tensor, controlnet_conditioning_scale=1, seed=-1):
    generator = torch.Generator("cuda").manual_seed(seed)    
    gen_img_1 = pipe(
        negative_prompt=negative_prompt,
        prompt=prompt,
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        num_inference_steps=20,
        image=control_tensor,
        generator=generator
    ).images[0]
    seed=1267
    gen_img_2 = pipe(
        negative_prompt=negative_prompt,
        prompt=prompt,
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        num_inference_steps=20,
        image=control_tensor,
        generator=generator
    ).images[0]
    return gen_img_1,gen_img_2

def process(input_image, prompt, negative_prompt):
    
    image = resize_image(input_image)
    mask = remove_bg_from_image(image)
    control_tensor = get_control_image_tensor(pipe.vae, image, mask)    
    
    gen_image_1,gen_image_2 = generate_(prompt, negative_prompt, control_tensor)

    result_image_1 = paste_fg_over_image(gen_image_1, image, mask)
    result_image_2 = paste_fg_over_image(gen_image_2, image, mask)

    return result_image_1,result_image_2


block = gr.Blocks().queue()

with block:
    gr.Markdown("## PHOT Background Replacer")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources='upload', type="pil", label="Upload", elem_id="image_upload", height=600)
            prompt = gr.Textbox(label="Prompt")
            negative_prompt = gr.Textbox(label="Negative prompt", value="Logo,Watermark,Text,Ugly,Morbid,Extra fingers,Poorly drawn hands,Mutation,Blurry,Extra limbs,Gross proportions,Missing arms,Mutated hands,Long neck,Duplicate,Mutilated,Mutilated hands,Poorly drawn face,Deformed,Bad anatomy,Cloned face,Malformed limbs,Missing legs,Too many fingers")
            # num_steps = gr.Slider(label="Number of steps", minimum=10, maximum=100, value=30, step=1)
            # controlnet_conditioning_scale = gr.Slider(label="Conditionig Scale", minimum=0.1, maximum=2.0, value=1.0, step=0.05)
            # seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, randomize=True)
            # seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, randomize=True,)
            run_button = gr.Button(value="Generate")
            
        with gr.Column():
            result_image_1 = gr.Image(label='Output Image 1', type="pil", show_label=True, elem_id="output-img-1")
            result_image_2 = gr.Image(label='Output Image 2', type="pil", show_label=True, elem_id="output-img-2")

    inputs = [input_image, prompt, negative_prompt]
    outputs = [result_image_1, result_image_2]
    run_button.click(fn=process, inputs=inputs, outputs=outputs)

block.launch(debug=True, share=True)