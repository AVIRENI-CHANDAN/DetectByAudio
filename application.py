import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from scipy.io.wavfile import write
from ultralytics import YOLO

from util import (
    AudioRecordConfiguration,
    CommandTextPosition,
    RecordBoxBoudaries,
    VideoCaptureWindowDimensions,
    YoloModels,
)


class ObjectDetectionApp:
    def __init__(self):
        # Initialize YOLOv8 model
        self.model = YOLO(YoloModels.MODEL)

        self.classes_names = self.model.names

        self.set_video_capture_dimension(0)

        self.fs = AudioRecordConfiguration.AUDIO_RATE
        self.seconds = AudioRecordConfiguration.DURATION
        self.audio_file_name = "output.wav"

        self.button_clicked = False
        self.command = ""

        # Create OpenCV window and set mouse callback
        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", self.record_audio_by_mouse_click)

    def set_video_capture_dimension(self, video_capture_mode):
        self.cap = cv2.VideoCapture(video_capture_mode)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VideoCaptureWindowDimensions.WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VideoCaptureWindowDimensions.HEIGHT)

    def record_audio_by_mouse_click(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if (
                RecordBoxBoudaries.x_start <= x <= RecordBoxBoudaries.x_end
                and RecordBoxBoudaries.y_start <= y <= RecordBoxBoudaries.y_end
            ):
                print("Click inside the button")
                self.record_audio()
                self.command = self.get_text_from_audio()

                if not self.button_clicked:
                    self.button_clicked = True
            else:
                print("Click outside the button")
                self.button_clicked = False

    def record_audio(self):
        print("Recording audio for 3 seconds...")
        myrecording = sd.rec(
            int(self.seconds * self.fs), samplerate=self.fs, channels=2
        )
        sd.wait()
        write(self.audio_file_name, self.fs, myrecording)

    def get_text_from_audio(self):
        data, samplerate = sf.read(self.audio_file_name)
        sf.write("outputNew.wav", data, samplerate, subtype="PCM_16")

        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile("outputNew.wav") as source:
                audio = recognizer.record(source)
                result = recognizer.recognize_google(audio)
                print("Recognized Text:", result)
                return result
        except sr.UnknownValueError:
            print("Speech recognition could not understand the audio.")
        except sr.RequestError as e:
            print(
                f"Error requesting results from Google Speech Recognition service: {e}"
            )
        return ""

    def detect_objects(self):
        while True:
            rtn, frame = self.cap.read()
            if not rtn:
                break

            results = self.model.predict(frame, stream=True, verbose=False)
            cv2.putText(
                frame,
                f"Recognised text: {self.command}",
                (CommandTextPosition.x, CommandTextPosition.y),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 0, 0),
                2,
            )

            # Process detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Get class name and confidence
                    class_id = int(box.cls[0])
                    name = self.classes_names[class_id]
                    # Draw detection if it matches the recognized command
                    if (
                        self.button_clicked
                        and self.command.lower().find(name.lower()) >= 0
                    ):
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 200), 3)
                        conf = float(box.conf[0])

                        cv2.putText(
                            frame,
                            f"{name} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1,
                            (50, 50, 220),
                            2,
                        )

            # Draw the "Record" button
            cv2.rectangle(
                frame,
                (RecordBoxBoudaries.x_start, RecordBoxBoudaries.y_start),
                (RecordBoxBoudaries.x_end, RecordBoxBoudaries.y_end),
                (0, 0, 153),
                0,
            )
            cv2.putText(
                frame,
                "Record for 3 seconds",
                (40, 60),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (25, 25, 25),
                2,
            )

            # Display the frame
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        self.detect_objects()
