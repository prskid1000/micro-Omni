# Chapter 40: Inference Examples

[← Previous: Running Training](39-running-training.md) | [Back to Index](00-INDEX.md) | [Next: Customization →](41-customization-guide.md)

---

## 🎯 Using Trained μOmni

Practical examples of using μOmni for different tasks.

---

## 💬 Example 1: Text Chat

```python
from omni import load_model

# Load model
model = load_model('checkpoints/omni_sft_tiny/omni_final.pt')

# Simple chat
response = model.chat("What is machine learning?")
print(response)
# Output: "Machine learning is a field of AI that..."

# Multi-turn conversation
conversation = []
while True:
    user_input = input("You: ")
    response = model.chat(user_input, history=conversation)
    print(f"Assistant: {response}")
    conversation.append({"user": user_input, "assistant": response})
```

---

## 🖼️ Example 2: Image Understanding

```python
# Image question answering
response = model.chat(
    text="What animal is in this image?",
    image="examples/cat.jpg"
)
print(response)  # "This is a cat"

# Image captioning
caption = model.chat(
    text="Describe this image in detail",
    image="examples/landscape.jpg"
)
print(caption)
```

---

## 🎤 Example 3: Audio Transcription

```python
# Transcribe audio
transcription = model.chat(
    text="Transcribe this audio",
    audio="examples/hello.wav"
)
print(transcription)  # "hello world"

# Audio + visual
response = model.chat(
    text="What do you see and hear?",
    image="examples/dog.jpg",
    audio="examples/bark.wav"
)
print(response)  # "I see a dog and hear it barking"
```

---

## 🔊 Example 4: Text-to-Speech

```python
# Generate speech from text
audio_path = model.generate_speech(
    text="Hello world, how are you?",
    output_path="output.wav"
)
print(f"Saved to: {audio_path}")

# You can play it:
import sounddevice as sd
import soundfile as sf
data, sr = sf.read('output.wav')
sd.play(data, sr)
```

---

## 🚀 Example 5: Batch Processing

```python
# Process multiple inputs
results = model.chat_batch([
    {"text": "What is AI?"},
    {"text": "Describe this", "image": "cat.jpg"},
    {"text": "Transcribe", "audio": "hello.wav"}
])

for i, result in enumerate(results):
    print(f"Result {i+1}: {result}")
```

---

## 🎛️ Advanced: Controlling Generation

```python
# Control generation parameters
response = model.chat(
    text="Write a story about a cat",
    max_tokens=200,      # Length
    temperature=0.8,     # Randomness (0=deterministic, 1=creative)
    top_p=0.9,           # Nucleus sampling
    repetition_penalty=1.2  # Reduce repetition
)
```

---

## 💡 Tips

✅ **Preprocess images** to 224×224 for best results  
✅ **Use 16kHz audio** for transcription  
✅ **Lower temperature** (0.3-0.5) for factual responses  
✅ **Higher temperature** (0.8-1.0) for creative tasks  
✅ **Enable KV caching** for faster generation

---

[Continue to Chapter 41: Customization Guide →](41-customization-guide.md)

---
