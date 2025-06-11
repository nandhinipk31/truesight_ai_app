# convert_to_state_dict.py
import torch
import torchvision.models as models
import torch.nn as nn

# Load the full model (only if you trust the file)
model = torch.load("models/ai_detector.pth", map_location=torch.device('cpu'), weights_only=False)
model.eval()

# Save only the state_dict
torch.save(model.state_dict(), "models/ai_detector_state_dict.pth")
print("✅ Model converted to state_dict and saved.")
