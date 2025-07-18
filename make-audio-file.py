import pyttsx3

engine = pyttsx3.init()
#engine.save_to_file("Hello, how are you doing today?", "your_audio.wav")
engine.save_to_file("안녕, 너는 뭐니 가나다라마바사아자차차차차차차","your_audio3.wav")
#engine.save_to_file("In this chapter, we first discussed the characteristics of a good crawler: scalability, politeness,extensibility, and robustness. Then, we proposed a design and discussed key components.Building a scalable web crawler is not a trivial task because the web is enormously large andfull of traps. Even though we have covered many topics, we still miss many relevant talking points:","your_audio2.wav")

engine.runAndWait()