import os

file_a = "/home/derrick/Documents/Wander Whisper/Wander-Whisper/src/model/evaluate.py"
file_b = "/home/derrick/Documents/Wander Whisper/Wander-Whisper/fine_tuned_models/fine_tuned_t5_small_travel_3_epochs"

relative_path = os.path.relpath(file_b, os.path.dirname(file_a))
print(relative_path)
