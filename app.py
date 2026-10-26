import gradio as gr
from src.inference import load_model, predict


model = load_model()


def segment_image(image):
    if image is None:
        raise gr.Error("Please upload an image.")

    return predict(model, image)


with gr.Blocks(title="Polyp Segmentation") as demo:
    gr.Markdown(
        "# Polyp Segmentation\n"
        "U-Net with a pretrained ResNet-18 encoder.\n\n"
        "Upload an endoscopy image to view its predicted "
        "polyp mask and overlay.\n\n"
        "**Research demonstration only — not for clinical use.**"
    )

    input_image = gr.Image(
        type="pil",
        sources=["upload"],
        label="Input image",
    )

    run_button = gr.Button("Segment image", variant="primary")

    with gr.Row():
        mask_output = gr.Image(label="Predicted mask")
        overlay_output = gr.Image(label="Segmentation overlay")

    run_button.click(
        fn=segment_image,
        inputs=input_image,
        outputs=[mask_output, overlay_output],
    )


if __name__ == "__main__":
    demo.launch()
