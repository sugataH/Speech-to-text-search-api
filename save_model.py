from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load pretrained model and processor from Hugging Face
model_name = "facebook/wav2vec2-base-960h"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

# Save both to your project folder
save_path = "C:/voice_search_project_2/model"
model.save_pretrained(save_path)
processor.save_pretrained(save_path)

print(f"Model and processor saved to: {save_path}")
