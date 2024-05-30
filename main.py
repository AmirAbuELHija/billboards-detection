import gradio as gr
import cv2
from ultralytics import YOLO
import torch
from simple_lama_inpainting import SimpleLama
from PIL import Image, ImageEnhance
import numpy as np
from scipy.ndimage import binary_fill_holes


def predict_and_inpaint(
        input_image,
        image_model_type,
        image_contrast,
        image_sharping,
        image_color,
        image_size,
):
    img = cv2.imread(input_image)
    img = cv2.resize(img, (image_size, image_size))
    model = YOLO(image_model_type)
    results = model.predict(source=img.copy(), save=True, save_txt=False, stream=True,
                            project='runs/segment', name='predict')
    binary_masks_list = []
    for result in results:
        if result.masks is None:
            raise gr.Error("Cannot detect a billboard in the image!")
        masks = result.masks.data
        boxes = result.boxes.data
        clss = boxes[:, 5]
        num_classes = 6
        for i in range(num_classes):
            if torch.any(torch.eq(clss, i)):
                people_indices = torch.where(clss == i)
                people_masks = masks[people_indices]
                people_mask = torch.any(people_masks, dim=0).int() * 255
                # save to file
                cv2.imwrite(str(model.predictor.save_dir / ('binary_mask' + str(i) + '.jpg')),
                            people_mask.cpu().numpy())
                current_mask = cv2.imread(str(model.predictor.save_dir / ('binary_mask' + str(i) + '.jpg')))
                binary_masks_list.append(current_mask)
        # add masks
        h, w, c = binary_masks_list[0].shape
        result_append = np.full((h, w, c), (0, 0, 0), dtype=np.uint8)
        for mask in binary_masks_list:
            result_append = cv2.add(result_append, mask)
        cv2.imwrite(str(model.predictor.save_dir / 'binary_mask.jpg'), result_append)

    simple_lama = SimpleLama()
    mask_path = model.predictor.save_dir / 'binary_mask.jpg'
    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask)
    mask = cv2.resize(mask, (320, 320))
    mask = binary_fill_holes(mask)

    original_dtype = mask.dtype
    mask = mask.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.blur(mask, (8, 8))
    mask = mask.astype(original_dtype)

    mask = Image.fromarray(mask)
    mask.save("mask.jpg")
    # Convert the image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert the image to PIL Image
    pil_img = Image.fromarray(img)
    # Enhance the sharpness
    enhancer = ImageEnhance.Sharpness(pil_img)
    img_enhanced = enhancer.enhance(image_sharping)
    # Enhance the contrast
    enhancer = ImageEnhance.Contrast(img_enhanced)
    img_enhanced = enhancer.enhance(image_contrast)
    # Enhance the color
    enhancer = ImageEnhance.Color(img_enhanced)
    img_enhanced = enhancer.enhance(image_color)
    img_enhanced = np.array(img_enhanced)
    img_enhanced = cv2.bilateralFilter(img_enhanced, d=7, sigmaColor=55, sigmaSpace=55)
    img_enhanced = cv2.resize(img_enhanced, (320, 320))
    img_enhanced = Image.fromarray(img_enhanced)

    result = simple_lama(img_enhanced, mask)
    result.save("output.png")
    result.save(str(model.predictor.save_dir / 'output.png'))
    mask_path = model.predictor.save_dir / 'image0.jpg'
    return [str(mask_path), "mask.jpg", "output.png"]


def image_app():
    with gr.Blocks():
        with gr.Row():
            input_image = gr.Image(type="filepath").style(height=280)
            with gr.Column():
                image_size = gr.Slider(
                    minimum=0,
                    maximum=1600,
                    step=32,
                    value=640,
                    label="Image Size",
                )
                image_contrast = gr.Slider(
                    minimum=0,
                    maximum=5,
                    step=0.1,
                    value=1.1,
                    label="Contrast",
                )
                image_sharping = gr.Slider(
                    minimum=0,
                    maximum=5,
                    step=0.1,
                    value=3,
                    label="Sharpness",
                )
                image_color = gr.Slider(
                    minimum=0,
                    maximum=5,
                    step=0.1,
                    value=1.3,
                    label="Color",
                )

        with gr.Row():
            with gr.Column():
                image_model_type = gr.Dropdown(
                    choices=[
                        "my-yolov8n-seg.pt",
                        "my-yolov8s-seg.pt",
                        "my-yolov8l-seg.pt",
                        "my-yolov8x-seg.pt",
                    ],
                    value="my-yolov8m-seg.pt",
                    label="Model Type",
                )
        image_predict = gr.Button(value="Run")
        with gr.Row():
            segmentation_mask = gr.Image(interactive=False, label="Segmentation Mask")
            binary_mask = gr.Image(interactive=False, label="Binary Mask")
            output_image = gr.Image(interactive=False, label="Output Image")

        image_predict.click(
            fn=predict_and_inpaint,
            inputs=[
                input_image,
                image_model_type,
                image_contrast,
                image_sharping,
                image_color,
                image_size,
            ],
            outputs=[segmentation_mask,
                     binary_mask,
                     output_image],
        )


def my_app():
    app = gr.Blocks()
    with app:
        gr.Markdown("# Billboard Removal Mini-Project")
        with gr.Row():
            with gr.Column():
                image_app()

    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True)


if __name__ == "__main__":
    my_app()
