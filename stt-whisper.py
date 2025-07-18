import whisper

model = whisper.load_model("base")
result = model.transcribe("2.mp3")
print("STT 결과:", result["text"])
