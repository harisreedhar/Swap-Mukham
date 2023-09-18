import os

FFMPEG_PATH = "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else None

FP16 = False

def set_fp16(value):
    global FP16
    FP16 = value

INSWAPPER_PATH = (
    "./assets/pretrained_models/inswapper_128.fp16.onnx"
    if FP16
    else "./assets/pretrained_models/inswapper_128.onnx"
)
FACE_PARSER_PATH = "./assets/pretrained_models/faceparser.onnx"
ARCFACE_PATH = "./assets/pretrained_models/w600k_r50.onnx"
RETINAFACE_PATH = "./assets/pretrained_models/det_10g.onnx"
OPEN_NSFW_PATH = "./assets/pretrained_models/open-nsfw.onnx"
GENDERAGE_PATH = "./assets/pretrained_models/gender_age.onnx"

CODEFORMER_PATH = "./assets/pretrained_models/codeformer.onnx"
GFPGAN_V14_PATH = "./assets/pretrained_models/GFPGANv1.4.onnx"
GFPGAN_V13_PATH = "./assets/pretrained_models/GFPGANv1.3.onnx"
GFPGAN_V12_PATH = "./assets/pretrained_models/GFPGANv1.2.onnx"
GPEN_BFR_512_PATH = "./assets/pretrained_models/GPEN-BFR-512.onnx"
GPEN_BFR_256_PATH = "./assets/pretrained_models/GPEN-BFR-256.onnx"
RESTOREFORMER_PATH = "./assets/pretrained_models/restoreformer.onnx"