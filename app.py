print(
"""
 ▄▄▄       ██▓ ███▄    █ ▓█████▄▄▄█████▓  ██████ ▓█████▄      ▄████  ██▀███   ▒█████   █    ██  ██▓███  
▒████▄    ▓██▒ ██ ▀█   █ ▓█   ▀▓  ██▒ ▓▒▒██    ▒ ▒██▀ ██▌    ██▒ ▀█▒▓██ ▒ ██▒▒██▒  ██▒ ██  ▓██▒▓██░  ██▒
▒██  ▀█▄  ▒██▒▓██  ▀█ ██▒▒███  ▒ ▓██░ ▒░░ ▓██▄   ░██   █▌   ▒██░▄▄▄░▓██ ░▄█ ▒▒██░  ██▒▓██  ▒██░▓██░ ██▓▒
░██▄▄▄▄██ ░██░▓██▒  ▐▌██▒▒▓█  ▄░ ▓██▓ ░   ▒   ██▒░▓█▄   ▌   ░▓█  ██▓▒██▀▀█▄  ▒██   ██░▓▓█  ░██░▒██▄█▓▒ ▒
 ▓█   ▓██▒░██░▒██░   ▓██░░▒████▒ ▒██▒ ░ ▒██████▒▒░▒████▓    ░▒▓███▀▒░██▓ ▒██▒░ ████▓▒░▒▒█████▓ ▒██▒ ░  ░
 ▒▒   ▓▒█░░▓  ░ ▒░   ▒ ▒ ░░ ▒░ ░ ▒ ░░   ▒ ▒▓▒ ▒ ░ ▒▒▓  ▒     ░▒   ▒ ░ ▒▓ ░▒▓░░ ▒░▒░▒░ ░▒▓▒ ▒ ▒ ▒▓▒░ ░  ░
  ▒   ▒▒ ░ ▒ ░░ ░░   ░ ▒░ ░ ░  ░   ░    ░ ░▒  ░ ░ ░ ▒  ▒      ░   ░   ░▒ ░ ▒░  ░ ▒ ▒░ ░░▒░ ░ ░ ░▒ ░     
  ░   ▒    ▒ ░   ░   ░ ░    ░    ░      ░  ░  ░   ░ ░  ░    ░ ░   ░   ░░   ░ ░ ░ ░ ▒   ░░░ ░ ░ ░░       
      ░  ░ ░           ░    ░  ░              ░     ░             ░    ░         ░ ░     ░              
                                                  ░                                                     
"""
)
import os
import cv2
import time
import shutil
import base64
import datetime
import argparse
import numpy as np
import gradio as gr
from tqdm import tqdm
import concurrent.futures

import threading
import default_paths as dp
cv_reader_lock = threading.Lock()

## ------------------------------ USER ARGS ------------------------------

parser = argparse.ArgumentParser(description="Swap-Mukham Face Swapper")
parser.add_argument("--out_dir", help="Default Output directory", default=os.getcwd())
parser.add_argument("--max_threads", type=int, help="Max num of threads to use", default=2)
parser.add_argument("--colab", action="store_true", help="Colab mode", default=False)
parser.add_argument("--cpu", action="store_true", help="Enable cpu mode", default=False)
parser.add_argument("--autolaunch", help="Auto start in browser", default=False, action="store_true")
parser.add_argument("--fp16", help="Use fp16 model Inswapper", default=False, action="store_true")
parser.add_argument("--prefer_text_widget", action="store_true", help="Replaces target video widget with text widget", default=False)
user_args = parser.parse_args()

USE_CPU = user_args.cpu

import default_paths as dp
import global_variables as gv

from swap_mukham import SwapMukham

from face_parsing import mask_regions_to_list

from utils.device import get_device_and_provider, device_types_list
from utils.image import (
    image_mask_overlay,
    resize_image_by_resolution,
    resolution_map,
    fast_pil_encode,
    fast_numpy_encode,
    get_crf_for_resolution,
)
from utils.io import (
    open_directory,
    get_images_from_directory,
    copy_files_to_directory,
    create_directory,
    get_single_video_frame,
    ffmpeg_merge_frames,
    ffmpeg_mux_audio,
    add_datetime_to_filename,
)

gr.processing_utils.encode_pil_to_base64 = fast_pil_encode
gr.processing_utils.encode_array_to_base64 = fast_numpy_encode

dp.set_fp16(user_args.fp16)

gv.USE_COLAB = user_args.colab
gv.MAX_THREADS = user_args.max_threads
gv.DEFAULT_OUTPUT_PATH = user_args.out_dir

PREFER_TEXT_WIDGET = user_args.prefer_text_widget

WORKSPACE = None
OUTPUT_FILE = None

#preferred_device = "cpu" if USE_CPU else "cuda"
preferred_device = "cuda"
DEVICE_LIST = device_types_list
DEVICE, PROVIDER, OPTIONS = get_device_and_provider(device=preferred_device)
SWAP_MUKHAM = SwapMukham(device=DEVICE)

IS_RUNNING = False
CURRENT_FRAME = None
COLLECTED_FACES = []
FOREGROUND_MASK_DICT = {}
NSFW_CACHE = {}


## ------------------------------ MAIN PROCESS ------------------------------


def process(
    test_mode,
    target_type,
    image_path,
    video_path,
    directory_path,
    source_path,
    use_foreground_mask,
    img_fg_mask,
    fg_mask_softness,
    output_path,
    output_name,
    use_datetime_suffix,
    sequence_output_format,
    keep_output_sequence,
    swap_condition,
    age,
    distance,
    face_enhancer_name,
    face_upscaler_opacity,
    use_face_parsing,
    parse_from_target,
    mask_regions,
    mask_blur_amount,
    mask_erode_amount,
    swap_iteration,
    face_scale,
    use_laplacian_blending,
    crop_top,
    crop_bott,
    crop_left,
    crop_right,
    current_idx,
    number_of_threads,
    use_frame_selection,
    frame_selection_ranges,
    video_quality,
    face_detection_condition,
    face_detection_size,
    face_detection_threshold,
    progress=gr.Progress(track_tqdm=True),
    *specifics,
):
    global WORKSPACE
    global OUTPUT_FILE
    global PREVIEW
    WORKSPACE, OUTPUT_FILE, PREVIEW = None, None, None

    global IS_RUNNING
    IS_RUNNING = True

    ## ------------------------------ GUI UPDATE FUNC ------------------------------
    def ui_before():
        return (
            gr.update(visible=True, value=None),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(visible=False, value=None),
        )

    def ui_after():
        return (
            gr.update(visible=True, value=PREVIEW),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(visible=False, value=None),
        )

    def ui_after_vid():
        return (
            gr.update(visible=False),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(value=OUTPUT_FILE, visible=True),
        )

    if not test_mode:
        yield ui_before()  # resets ui preview
        progress(0, desc="Обработка")

    start_time = time.time()
    total_exec_time = lambda start_time: divmod(time.time() - start_time, 60)
    get_finsh_text = (
        lambda start_time: f"Завершено за {int(total_exec_time(start_time)[0])} мин {int(total_exec_time(start_time)[1])} сек."
    )

    ## ------------------------------ PREPARE INPUTS ------------------------------

    if use_datetime_suffix:
        output_name = add_datetime_to_filename(output_name)

    mask_regions = mask_regions_to_list(mask_regions)

    specifics = list(specifics)
    half = len(specifics) // 2
    if swap_condition == "specific face":
        source_specifics = [
            (src, spc) for src, spc in zip(specifics[:half], specifics[half:])
        ]
    else:
        source_specifics = [(source_path, None)]

    if crop_top > crop_bott:
        crop_top, crop_bott = crop_bott, crop_top
    if crop_left > crop_right:
        crop_left, crop_right = crop_right, crop_left
    crop_mask = (crop_top, 511 - crop_bott, crop_left, 511 - crop_right)

    input_args = {
        "similarity": distance,
        "age": age,
        "face_scale": face_scale,
        "num_of_pass": swap_iteration,
        "face_upscaler_opacity": face_upscaler_opacity,
        "mask_crop_values": crop_mask,
        "mask_erode_amount": mask_erode_amount,
        "mask_blur_amount": mask_blur_amount,
        "use_laplacian_blending": use_laplacian_blending,
        "swap_condition": swap_condition,
        "face_parse_regions": mask_regions,
        "use_face_parsing": use_face_parsing,
        "face_detection_size": [int(face_detection_size), int(face_detection_size)],
        "face_detection_threshold": face_detection_threshold,
        "face_detection_condition": face_detection_condition,
        "parse_from_target": parse_from_target
    }

    SWAP_MUKHAM.set_values(input_args)
    if (
        SWAP_MUKHAM.face_upscaler is None
        or SWAP_MUKHAM.face_upscaler_name != face_enhancer_name
    ):
        SWAP_MUKHAM.load_face_upscaler(face_enhancer_name, device=DEVICE)
    if SWAP_MUKHAM.face_parser is None and use_face_parsing:
        SWAP_MUKHAM.load_face_parser(device=DEVICE)
    SWAP_MUKHAM.analyse_source_faces(source_specifics)

    mask = None
    if use_foreground_mask and img_fg_mask is not None:
        mask = img_fg_mask.get("mask", None)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2RGB)
        if fg_mask_softness > 0:
            mask = cv2.blur(mask, (int(fg_mask_softness), int(fg_mask_softness)))
        mask = mask.astype("float32") / 255.0


    ## ------------------------------ IMAGE ------------------------------

    if target_type == "Image" and not test_mode:
        target = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        output = SWAP_MUKHAM.process_frame(
            [target, mask]
        )
        output_file = os.path.join(output_path, output_name + ".png")
        cv2.imwrite(output_file, output)

        PREVIEW = output
        OUTPUT_FILE = output_file
        WORKSPACE = output_path

        gr.Info(get_finsh_text(start_time))
        yield ui_after()

    ## ------------------------------ VIDEO ------------------------------

    elif target_type == "Video" and not test_mode:
        video_path = video_path.replace('"', '').strip()

        temp_path = os.path.join(output_path, output_name)
        os.makedirs(temp_path, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        is_in_range = lambda idx: any([int(rng[0]) <= idx <= int(rng[1]) for rng in frame_selection_ranges]) if use_frame_selection else True

        print("[ Процесс замены лиц(а) начался ]")

        def swap_video_func(frame_index):
            if IS_RUNNING:
                with cv_reader_lock:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                    valid_frame, frame = cap.read()

                if valid_frame:
                    if is_in_range(frame_index):
                        mask = FOREGROUND_MASK_DICT.get(frame_index, None) if use_foreground_mask else None
                        output = SWAP_MUKHAM.process_frame([frame, mask])
                    else:
                        output = frame
                    frame_path = os.path.join(temp_path, f"frame_{frame_index}.{sequence_output_format}")
                    if sequence_output_format == "jpg":
                        cv2.imwrite(frame_path, output, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    else:
                        cv2.imwrite(frame_path, output)

        with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_threads) as executor:
            futures = [executor.submit(swap_video_func, idx) for idx in range(total_frames)]

            with tqdm(total=total_frames, desc="Processing") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    pbar.update(1)

        cap.release()

        if IS_RUNNING:
            print("[ Объединение изображений ]")
            progress(0, desc="Объединение изображений")
            WORKSPACE = output_path
            out_without_audio = output_name + "_without_audio" + ".mp4"
            destination = os.path.join(output_path, out_without_audio)
            crf = get_crf_for_resolution(max(width,height), video_quality)
            ret, destination = ffmpeg_merge_frames(
                temp_path, f"frame_%d.{sequence_output_format}", destination, fps=fps, crf=crf, ffmpeg_path=dp.FFMPEG_PATH
            )
            OUTPUT_FILE = destination

            if ret:
                print("[ Объединение аудио ]")
                progress(0, desc="Объединение аудио")
                OUTPUT_FILE = destination
                out_with_audio = out_without_audio.replace("_without_audio", "")
                _ret, _destination = ffmpeg_mux_audio(
                    video_path, out_without_audio, out_with_audio, ffmpeg_path=dp.FFMPEG_PATH
                )

                if _ret:
                    OUTPUT_FILE = _destination
                    os.remove(out_without_audio)

            if os.path.exists(temp_path) and not keep_output_sequence:
                print("[ Удаление временных файлов ]")
                progress(0, desc="Удаление временных файлов")
                shutil.rmtree(temp_path)

            finish_text = get_finsh_text(start_time)
            print(f"[ {finish_text} ]")
            gr.Info(finish_text)
            yield ui_after_vid()

    ## ------------------------------ DIRECTORY ------------------------------

    elif target_type == "Directory" and not test_mode:
        temp_path = os.path.join(output_path, output_name)
        temp_path = create_directory(temp_path, remove_existing=True)

        directory_path = directory_path.replace('"', '').strip()
        image_paths = get_images_from_directory(directory_path)

        new_image_paths = copy_files_to_directory(image_paths, temp_path)

        def swap_func(img_path):
            if IS_RUNNING:
                frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                output = SWAP_MUKHAM.process_frame([frame, None])
                cv2.imwrite(img_path, output)

        with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_threads) as executor:
            futures = [executor.submit(swap_func, img_path) for img_path in new_image_paths]

            with tqdm(total=len(new_image_paths), desc="Обработка") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    pbar.update(1)

        PREVIEW = cv2.imread(new_image_paths[-1])
        WORKSPACE = temp_path
        OUTPUT_FILE = new_image_paths[-1]

        gr.Info(get_finsh_text(start_time))
        yield ui_after()

    ## ------------------------------ STREAM ------------------------------

    elif target_type == "Stream" and not test_mode:
        pass

    ## ------------------------------ TEST ------------------------------

    if test_mode and target_type == "Video":
        mask = None
        if use_face_parsing_mask:
            mask = FOREGROUND_MASK_DICT.get(current_idx, None)
        if CURRENT_FRAME is not None and isinstance(CURRENT_FRAME, np.ndarray):
            PREVIEW = SWAP_MUKHAM.process_frame(
                [CURRENT_FRAME[:, :, ::-1], mask]
            )
            gr.Info(get_finsh_text(start_time))
            yield ui_after()


## ------------------------------ GRADIO GUI ------------------------------

css = """

div.gradio-container{
    max-width: unset !important;
}

footer{
    display:none !important
}

#slider_row {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
}

#refresh_slider {
  flex: 0 1 20%;
  display: flex;
  align-items: center;
}

#frame_slider {
  flex: 1 0 80%;
  display: flex;
  align-items: center;
}

"""

WIDGET_PREVIEW_HEIGHT = 450

with gr.Blocks(css=css, theme=gr.themes.Default()) as interface:
    gr.Markdown("# 🗿 Swap Mukham")
    with gr.Row():
        with gr.Row():
            with gr.Column(scale=0.35):
                with gr.Tabs():
                    with gr.TabItem("📄 Ввод"):
                        swap_condition = gr.Dropdown(
                            gv.FACE_DETECT_CONDITIONS,
                            info="Выберите, какое лицо или лица которое заменяем нужно поменять местами.",
                            multiselect=False,
                            show_label=False,
                            value=gv.FACE_DETECT_CONDITIONS[0],
                            interactive=True,
                        )
                        age = gr.Number(
                            value=25, label="Значение", interactive=True, visible=False
                        )

                        ## ------------------------------ SOURCE IMAGE ------------------------------

                        source_image_input = gr.Image(
                            label="Лицо на которое будет заменено", type="filepath", interactive=True, height=200,
                        )

                        ## ------------------------------ SOURCE SPECIFIC ------------------------------

                        with gr.Box(visible=False) as specific_face:
                            for i in range(gv.NUM_OF_SRC_SPECIFIC):
                                idx = i + 1
                                code = "\n"
                                code += f"with gr.Tab(label='{idx}'):"
                                code += "\n\twith gr.Row():"
                                code += f"\n\t\tsrc{idx} = gr.Image(interactive=True, type='numpy', label='Лицо на которое будет заменено {idx}', height=200)"
                                code += f"\n\t\ttrg{idx} = gr.Image(interactive=True, type='numpy', label='Лицо которое заменяем {idx}', height=200)"
                                exec(code)

                            distance_slider = gr.Slider(
                                minimum=0,
                                maximum=2,
                                value=0.6,
                                interactive=True,
                                label="Расстояние",
                                info="Меньшее расстояние более похоже, а большее расстояние менее похоже на лицо которое заменяем.",
                            )

                        ## ------------------------------ TARGET TYPE ------------------------------

                        with gr.Group():
                            target_type = gr.Radio(
                                ["Image", "Video", "Directory"],
                                label="С чем будем работать",
                                value="Video",
                            )

                            ## ------------------------------ TARGET IMAGE ------------------------------

                            with gr.Box(visible=False) as input_image_group:
                                target_image_input = gr.Image(
                                    label="Лицо которое заменяем",
                                    interactive=True,
                                    type="filepath",
                                    height=200
                                )

                            ## ------------------------------ TARGET VIDEO ------------------------------

                            with gr.Box(visible=True) as input_video_group:
                                with gr.Column():
                                    video_widget = gr.Text if PREFER_TEXT_WIDGET else gr.Video
                                    video_input = video_widget(
                                        label="Видео на котором заменяем", interactive=True,
                                    )

                                    ## ------------------------------ FRAME SELECTION ------------------------------

                                    with gr.Accordion("Выбор кадра", open=False):
                                        use_frame_selection = gr.Checkbox(
                                            label="Используйте выбор кадра", value=False, interactive=True,
                                        )
                                        frame_selection_ranges = gr.Numpy(
                                            headers=["Начальный кадр", "Конечный кадр"],
                                            datatype=["number", "number"],
                                            row_count=1,
                                            col_count=(2, "fixed"),
                                            interactive=True
                                        )

                            ## ------------------------------ TARGET DIRECTORY ------------------------------

                            with gr.Box(visible=False) as input_directory_group:
                                directory_input = gr.Text(
                                    label="Директория изображений на которых будем заменять", interactive=True
                                )

                    ## ------------------------------ TAB MODEL ------------------------------

                    with gr.TabItem("🎚️ Модель"):
                        with gr.Accordion("Замена", open=True):
                            with gr.Row():
                                swap_iteration = gr.Slider(
                                    label="Итераций замены",
                                    minimum=1,
                                    maximum=4,
                                    value=1,
                                    step=1,
                                    interactive=True,
                                )
                                face_scale = gr.Slider(
                                    label="Масштаб лица",
                                    minimum=0,
                                    maximum=2,
                                    value=1,
                                    interactive=True,
                                )
                        with gr.Accordion("Обнаружение", open=True):
                            face_detection_condition = gr.Dropdown(
                                gv.SINGLE_FACE_DETECT_CONDITIONS,
                                label="Условие",
                                value=gv.DETECT_CONDITION,
                                interactive=True,
                                info="Это условие используется только в том случае, если на (лицо на которое заменяем) или (лицо которое заменяем) обнаружено несколько лиц.",
                            )
                            face_detection_size = gr.Number(
                                label="Размер обнаружения",
                                value=gv.DETECT_SIZE,
                                interactive=True,
                            )
                            face_detection_threshold = gr.Number(
                                label="Порог обнаружения",
                                value=gv.DETECT_THRESHOLD,
                                interactive=True,
                            )

                    ## ------------------------------ TAB POST-PROCESS ------------------------------

                    with gr.TabItem("🪄 Постобработка"):
                        with gr.Row():
                            face_enhancer_name = gr.Dropdown(
                                gv.FACE_ENHANCER_LIST,
                                label="Восстановление лиц",
                                value="NONE",
                                multiselect=False,
                                interactive=True,
                            )
                            face_upscaler_opacity = gr.Slider(
                                label="Влияние",
                                minimum=0,
                                maximum=1,
                                value=1,
                                step=0.001,
                                interactive=True,
                            )

                        with gr.Accordion("Лицевая маска", open=False):
                            with gr.Group():
                                with gr.Row():
                                    use_face_parsing_mask = gr.Checkbox(
                                        label="Включить анализ лиц",
                                        value=False,
                                        interactive=True,
                                    )
                                    parse_from_target = gr.Checkbox(
                                        label="Анализировать из (Лицо которое заменяем)",
                                        value=False,
                                        interactive=True,
                                    )
                                mask_regions = gr.Dropdown(
                                    gv.MASK_REGIONS,
                                    value=gv.MASK_REGIONS_DEFAULT,
                                    multiselect=True,
                                    label="Включить",
                                    interactive=True,
                                )

                        with gr.Accordion("Ограничительная рамка «Обрезка лица»", open=False):
                            with gr.Group():
                                with gr.Row():
                                    crop_top = gr.Slider(
                                        label="Верх",
                                        minimum=0,
                                        maximum=511,
                                        value=0,
                                        step=1,
                                        interactive=True,
                                    )
                                    crop_bott = gr.Slider(
                                        label="Низ",
                                        minimum=0,
                                        maximum=511,
                                        value=511,
                                        step=1,
                                        interactive=True,
                                    )
                                with gr.Row():
                                    crop_left = gr.Slider(
                                        label="Лево",
                                        minimum=0,
                                        maximum=511,
                                        value=0,
                                        step=1,
                                        interactive=True,
                                    )
                                    crop_right = gr.Slider(
                                        label="Право",
                                        minimum=0,
                                        maximum=511,
                                        value=511,
                                        step=1,
                                        interactive=True,
                                    )

                        with gr.Row():
                            mask_erode_amount = gr.Slider(
                                label="Смешение лиц",
                                minimum=0,
                                maximum=1,
                                value=gv.MASK_ERODE_AMOUNT,
                                step=0.001,
                                interactive=True,
                            )

                            mask_blur_amount = gr.Slider(
                                label="Блюр Маски",
                                minimum=0,
                                maximum=1,
                                value=gv.MASK_BLUR_AMOUNT,
                                step=0.001,
                                interactive=True,
                            )

                        use_laplacian_blending = gr.Checkbox(
                            label="Лапласово смешивание",
                            value=True,
                            interactive=True,
                        )

                    ## ------------------------------ TAB OUTPUT ------------------------------

                    with gr.TabItem("📤 Вывод"):
                        output_directory = gr.Text(
                            label="Директория вывода",
                            value=gv.DEFAULT_OUTPUT_PATH,
                            interactive=True,
                        )
                        with gr.Group():
                            output_name = gr.Text(
                                label="Имя результата", value="Result", interactive=True
                            )
                            use_datetime_suffix = gr.Checkbox(
                                label="Суффикс даты и времени", value=True, interactive=True
                            )
                        with gr.Accordion("Настройки видео", open=True):
                            with gr.Row():
                                sequence_output_format = gr.Dropdown(
                                        ["jpg", "png"],
                                        label="Формат",
                                        value="jpg",
                                        interactive=True,
                                    )
                                video_quality = gr.Dropdown(
                                    gv.VIDEO_QUALITY_LIST,
                                    label="Качество",
                                    value=gv.VIDEO_QUALITY,
                                    interactive=True
                                )
                            keep_output_sequence = gr.Checkbox(
                                label="Сохранять последовательность вывода", value=False, interactive=True
                            )

                    ## ------------------------------ TAB PERFORMANCE ------------------------------
                    with gr.TabItem("🛠️ Производительность"):
                        preview_resolution = gr.Dropdown(
                            gv.RESOLUTIONS,
                            label="Разрешение превью",
                            value="Original",
                            interactive=True,
                        )
                        number_of_threads = gr.Number(
                            step=1,
                            interactive=True,
                            label="Максимальное количество потоков",
                            value=gv.MAX_THREADS,
                            minimum=1,
                        )
                        with gr.Box():
                            with gr.Column():
                                with gr.Row():
                                    face_analyser_device = gr.Radio(
                                        DEVICE_LIST,
                                        label="Обнаружение и распознавание лиц",
                                        value=DEVICE,
                                        interactive=True,
                                    )
                                    face_analyser_device_submit = gr.Button("Применить")
                                with gr.Row():
                                    face_swapper_device = gr.Radio(
                                        DEVICE_LIST,
                                        label="Замена лиц",
                                        value=DEVICE,
                                        interactive=True,
                                    )
                                    face_swapper_device_submit = gr.Button("Применить")
                                with gr.Row():
                                    face_parser_device = gr.Radio(
                                        DEVICE_LIST,
                                        label="Парсинг лиц",
                                        value=DEVICE,
                                        interactive=True,
                                    )
                                    face_parser_device_submit = gr.Button("Применить")
                                with gr.Row():
                                    face_upscaler_device = gr.Radio(
                                        DEVICE_LIST,
                                        label="Восстановление лиц",
                                        value=DEVICE,
                                        interactive=True,
                                    )
                                    face_upscaler_device_submit = gr.Button("Применить")

                                face_analyser_device_submit.click(
                                    fn=lambda d: SWAP_MUKHAM.load_face_analyser(
                                        device=d
                                    ),
                                    inputs=[face_analyser_device],
                                )
                                face_swapper_device_submit.click(
                                    fn=lambda d: SWAP_MUKHAM.load_face_swapper(
                                        device=d
                                    ),
                                    inputs=[face_swapper_device],
                                )
                                face_parser_device_submit.click(
                                    fn=lambda d: SWAP_MUKHAM.load_face_parser(device=d),
                                    inputs=[face_parser_device],
                                )
                                face_upscaler_device_submit.click(
                                    fn=lambda n, d: SWAP_MUKHAM.load_face_upscaler(
                                        n, device=d
                                    ),
                                    inputs=[face_enhancer_name, face_upscaler_device],
                                )

            ## ------------------------------ SWAP, CANCEL, FRAME SLIDER ------------------------------

            with gr.Column(scale=0.65):
                with gr.Row():
                    swap_button = gr.Button("✨ Заменить", variant="primary")
                    cancel_button = gr.Button("⛔ Отмена")
                    collect_faces = gr.Button("👨 Обнаружить лица")
                    test_swap = gr.Button("🧪 Тестовая замена")

                with gr.Box() as frame_slider_box:
                    with gr.Row(elem_id="slider_row", equal_height=True):
                        set_slider_range_btn = gr.Button(
                            "Разбить на кадры", interactive=True, elem_id="refresh_slider"
                        )
                        frame_slider = gr.Slider(
                            label="Кадры",
                            minimum=0,
                            maximum=1,
                            value=0,
                            step=1,
                            interactive=True,
                            elem_id="frame_slider",
                        )

                ## ------------------------------ PREVIEW ------------------------------

                with gr.Tabs():
                    with gr.TabItem("Превью"):

                        preview_image = gr.Image(
                            label="Превью", type="numpy", interactive=False, height=WIDGET_PREVIEW_HEIGHT,
                        )

                        preview_video = gr.Video(
                            label="Вывод", interactive=False, visible=False, height=WIDGET_PREVIEW_HEIGHT,
                        )
                        preview_enabled_text = gr.Markdown(
                            "Отключить рисование переднего плана для превью!", visible=False
                        )
                        with gr.Row():
                            output_directory_button = gr.Button(
                                "📂", interactive=False, visible=not gv.USE_COLAB
                            )
                            output_video_button = gr.Button(
                                "🎬", interactive=False, visible=not gv.USE_COLAB
                            )

                            output_directory_button.click(
                                lambda: open_directory(path=WORKSPACE),
                                inputs=None,
                                outputs=None,
                            )
                            output_video_button.click(
                                lambda: open_directory(path=OUTPUT_FILE),
                                inputs=None,
                                outputs=None,
                            )

                    ## ------------------------------ FOREGROUND MASK ------------------------------

                    with gr.TabItem("Заменить передний план"):
                        with gr.Box() as fg_mask_group:
                            with gr.Row():
                                with gr.Row():
                                    use_foreground_mask = gr.Checkbox(
                                    label="Использовать маску переднего плана", value=False, interactive=True)
                                fg_mask_softness = gr.Slider(
                                    label="Мягкость Маски",
                                    minimum=0,
                                    maximum=200,
                                    value=1,
                                    step=1,
                                    interactive=True,
                                )
                                add_fg_mask_btn = gr.Button("Добить", interactive=True)
                                del_fg_mask_btn = gr.Button("Удалить", interactive=True)
                            img_fg_mask = gr.Image(
                                label="Рисовка маски",
                                tool="sketch",
                                interactive=True,
                                type="numpy",
                                height=WIDGET_PREVIEW_HEIGHT,
                            )

                    ## ------------------------------ COLLECT FACE ------------------------------

                    with gr.TabItem("Обнаруженные лица"):
                        collected_faces = gr.Gallery(
                            label="Лица",
                            show_label=False,
                            elem_id="gallery",
                            columns=[6], rows=[6], object_fit="contain", height=WIDGET_PREVIEW_HEIGHT,
                        )


    ## ------------------------------ GRADIO EVENTS ------------------------------

    def on_target_type_change(value):
        visibility = {
            "Image": (True, False, False, False, True, False, False, False),
            "Video": (False, True, False, True, True, True, True, True),
            "Directory": (False, False, True, False, False, False, False, False),
            "Stream": (False, False, True, False, False, False, False, False),
        }
        return list(gr.update(visible=i) for i in visibility[value])

    target_type.change(
        on_target_type_change,
        inputs=[target_type],
        outputs=[
            input_image_group,
            input_video_group,
            input_directory_group,
            frame_slider_box,
            fg_mask_group,
            add_fg_mask_btn,
            del_fg_mask_btn,
            test_swap,
        ],
    )

    target_image_input.change(
        lambda inp: gr.update(value=inp),
        inputs=[target_image_input],
        outputs=[img_fg_mask]
    )

    def on_swap_condition_change(value):
        visibility = {
            "age less than": (True, False, True),
            "age greater than": (True, False, True),
            "specific face": (False, True, False),
        }
        return tuple(
            gr.update(visible=i) for i in visibility.get(value, (False, False, True))
        )

    swap_condition.change(
        on_swap_condition_change,
        inputs=[swap_condition],
        outputs=[age, specific_face, source_image_input],
    )

    def on_set_slider_range(video_path):
        if video_path is None or not os.path.exists(video_path):
            gr.Info("Проверьте видео путь")
        else:
            try:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if total_frames > 0:
                    total_frames -= 1
                    return gr.Slider.update(
                        minimum=0, maximum=total_frames, value=0, interactive=True
                    )
                gr.Info("Ошибка при загрузке видео.")
            except:
                gr.Info("Ошибка при загрузке видео.")

    set_slider_range_event = set_slider_range_btn.click(
        on_set_slider_range,
        inputs=[video_input],
        outputs=[frame_slider],
    )

    def update_preview(video_path, frame_index, use_foreground_mask, resolution):
        if not os.path.exists(video_path):
            yield gr.update(value=None), gr.update(value=None), gr.update(visible=False)
        else:
            frame = get_single_video_frame(video_path, frame_index)
            if frame is not None:
                if use_foreground_mask:
                    overlayed_image = frame
                    if frame_index in FOREGROUND_MASK_DICT.keys():
                        mask = FOREGROUND_MASK_DICT.get(frame_index, None)
                        if mask is not None:
                            overlayed_image = image_mask_overlay(frame, mask)
                        yield gr.update(value=None), gr.update(value=None), gr.update(visible=False)  # clear previous mask
                    frame = resize_image_by_resolution(frame, resolution)
                    yield gr.update(value=frame[:, :, ::-1]), gr.update(
                        value=overlayed_image[:, :, ::-1], visible=True
                    ), gr.update(visible=False)
                else:
                    frame = resize_image_by_resolution(frame, resolution)
                    yield gr.update(value=frame[:, :, ::-1]), gr.update(value=None), gr.update(
                        visible=False
                    )

                global CURRENT_FRAME
                CURRENT_FRAME = frame

    frame_slider_event = frame_slider.change(
        fn=update_preview,
        inputs=[video_input, frame_slider, use_foreground_mask, preview_resolution],
        outputs=[preview_image, img_fg_mask, preview_video],
        show_progress=False,
    )

    def add_foreground_mask(fg, frame_index, softness):
        if fg is not None:
            mask = fg.get("mask", None)
            if mask is not None:
                alpha_rgb = cv2.cvtColor(mask, cv2.COLOR_BGRA2RGB)
                alpha_rgb = cv2.blur(alpha_rgb, (softness, softness))
                FOREGROUND_MASK_DICT[frame_index] = alpha_rgb.astype("float32") / 255.0
                gr.Info(f"индекс сохраненной маски {frame_index}")

    add_foreground_mask_event = add_fg_mask_btn.click(
        fn=add_foreground_mask,
        inputs=[img_fg_mask, frame_slider, fg_mask_softness],
    ).then(
        fn=update_preview,
        inputs=[video_input, frame_slider, use_foreground_mask, preview_resolution],
        outputs=[preview_image, img_fg_mask, preview_video],
        show_progress=False,
    )

    def delete_foreground_mask(frame_index):
        if frame_index in FOREGROUND_MASK_DICT.keys():
            FOREGROUND_MASK_DICT.pop(frame_index)
            gr.Info(f"Удален индекс маски {frame_index}")

    del_custom_mask_event = del_fg_mask_btn.click(
        fn=delete_foreground_mask, inputs=[frame_slider]
    ).then(
        fn=update_preview,
        inputs=[video_input, frame_slider, use_foreground_mask, preview_resolution],
        outputs=[preview_image, img_fg_mask, preview_video],
        show_progress=False,
    )

    def get_collected_faces(image):
        if image is not None:
            gr.Info(f"Обнурижение лиц...")
            faces = SWAP_MUKHAM.collect_heads(image)
            COLLECTED_FACES.extend(faces)
            yield COLLECTED_FACES
            gr.Info(f"Обнаружено {len(faces)} лиц")

    collect_faces.click(get_collected_faces, inputs=[preview_image], outputs=[collected_faces])

    src_specific_inputs = []
    gen_variable_txt = ",".join(
        [f"src{i+1}" for i in range(gv.NUM_OF_SRC_SPECIFIC)]
        + [f"trg{i+1}" for i in range(gv.NUM_OF_SRC_SPECIFIC)]
    )
    exec(f"src_specific_inputs = ({gen_variable_txt})")

    test_mode = gr.Checkbox(value=False, visible=False)

    swap_inputs = [
        test_mode,
        target_type,
        target_image_input,
        video_input,
        directory_input,
        source_image_input,
        use_foreground_mask,
        img_fg_mask,
        fg_mask_softness,
        output_directory,
        output_name,
        use_datetime_suffix,
        sequence_output_format,
        keep_output_sequence,
        swap_condition,
        age,
        distance_slider,
        face_enhancer_name,
        face_upscaler_opacity,
        use_face_parsing_mask,
        parse_from_target,
        mask_regions,
        mask_blur_amount,
        mask_erode_amount,
        swap_iteration,
        face_scale,
        use_laplacian_blending,
        crop_top,
        crop_bott,
        crop_left,
        crop_right,
        frame_slider,
        number_of_threads,
        use_frame_selection,
        frame_selection_ranges,
        video_quality,
        face_detection_condition,
        face_detection_size,
        face_detection_threshold,
        *src_specific_inputs,
    ]

    swap_outputs = [
        preview_image,
        output_directory_button,
        output_video_button,
        preview_video,
    ]

    swap_event = swap_button.click(fn=process, inputs=swap_inputs, outputs=swap_outputs)

    test_swap_settings = swap_inputs
    test_swap_settings[0] = gr.Checkbox(value=True, visible=False)

    test_swap_event = test_swap.click(
        fn=update_preview,
        inputs=[video_input, frame_slider, use_foreground_mask, preview_resolution],
        outputs=[preview_image, preview_video],
        show_progress=False,
    ).then(
        fn=process, inputs=test_swap_settings, outputs=swap_outputs, show_progress=True
    )

    def stop_running():
        global IS_RUNNING
        IS_RUNNING = False
        print("[ Процесс отменен ]")
        gr.Info("Процесс отменен")

    cancel_button.click(
        fn=stop_running,
        inputs=None,
        cancels=[swap_event, set_slider_range_event, test_swap_event],
        show_progress=True,
    )

def clear_temp_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            shutil.rmtree(os.path.join(root, dir))

folder = 'tmp/gradio'  # Замените на ваш путь к папке

clear_temp_folder(folder)

if __name__ == "__main__":
    if gv.USE_COLAB:
        print("Запуск в режиме colab")

    interface.queue(concurrency_count=2, max_size=20).launch(share=gv.USE_COLAB,inbrowser=user_args.autolaunch)
