import os
import torch
import torchaudio
from datetime import datetime
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import logging
import time
device = torch.cuda.is_available()
logger = logging.getLogger(__name__)
print("Loading model...")
config = XttsConfig()
config.load_json("E:\\ML\\ml_projects\\project_folder\\tts_datasets\\indictts\\hi\\ckpoints\\config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config,
                    checkpoint_dir="E:\\ML\\ml_projects\\project_folder\\tts_datasets\\indictts\\hi\\ckpoints\\",
                    use_deepspeed=False)
#config.enable_readaction=True
# model.cuda()
speakerpath = "E:\\ML\\ml_projects\\project_folder\\tts_datasets\\indictts\\hi\\speakers-hi\\"
phrases = ["मैं एक लड़का हूँ", "तुम आज बहुत सुंदर लग रही हो."]
print(len(phrases))
for filename in os.listdir(speakerpath):
    if filename.endswith(".wav"):
        for phrase in phrases:
            start_time = time.time()

            print("Computing speaker latents...")
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[speakerpath+filename])

            print("Inference...")
            out = model.inference(
            phrase,
            "hi",
            gpt_cond_latent, 
            speaker_embedding,
            temperature=0.7, # Add custom parameters here
        )
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # compute stats
            process_time = time.time() - start_time
            audio_time = len(torch.tensor(out["wav"]).unsqueeze(0) / 24000)
            logger.warning("Processing time: %.3f", process_time)
            logger.warning("Real-time factor: %.3f", process_time / audio_time)
            torchaudio.save(f"{now}-xtts.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)