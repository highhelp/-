import speech_recognition as sr
import numpy as np
import whisper


def get_speech(audio_path=None, mic=False):
    """파일 또는 마이크에서 소리 받아와 AudioData 객체로 반환하는 함수"""
    if audio_path is None and not mic:
        raise ValueError("You must specify an audio path or mic")

    recognizer = sr.Recognizer()

    if mic:
        # 마이크 설정
        microphone = sr.Microphone(sample_rate=16000)

        # 마이크 소음 수치 반영
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            print("소음 수치 반영하여 음성을 청취합니다. {}".format(recognizer.energy_threshold))

        # 음성 수집
        with microphone as source:
            print("목소리를 들을 준비가 되었습니다. 말씀해주세요 :)")
            audio = recognizer.listen(source)

        return audio

    # 파일에서 읽기
    audio_file = sr.AudioFile(audio_path)

    with audio_file as source:
        audio = recognizer.record(source)

    return audio


def whisper_stt(audio_data):
    """whisper를 이용한 STT"""
    audio_bytes = audio_data.get_raw_data()

    # 모댈에 입력 모양과 맞추기
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    model = whisper.load_model("base")
    result = model.transcribe(audio_np, fp16=False, language='ko')

    return result['text']


if __name__ == "__main__":
    import torch
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    audio_path = 'hello_16k.wav'

    audio_data = get_speech(audio_path=audio_path)
    result = whisper_stt(audio_data)
    print(result)