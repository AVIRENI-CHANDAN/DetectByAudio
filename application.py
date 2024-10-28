import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from scipy.io.wavfile import write
from util import (
    RecordBoxBoudaries,
    YoloConfigFiles,
    VideoCaptureWindowDimensions,
    AudioRecordConfiguration,
)


class ObjectDetectionApp:
    def __init__(self):
        self.net = cv2.dnn.readNet(
            YoloConfigFiles.WEIGHTS_FILE, YoloConfigFiles.CONFIG_FILE
        )
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1 / 255)

        self.classes_names = self.read_class_names()

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

    def read_class_names(self):
        classes = []
        with open("classes.txt", "r") as classes_file:
            classes = [class_name.strip() for class_name in classes_file.readlines()]
        return classes

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
        sd.wait()  # wait until the recording is finished
        write(self.audio_file_name, self.fs, myrecording)  # save the audio file

    def get_text_from_audio(self):
        # Convert the audio file for Google API
        data, samplerate = sf.read(self.audio_file_name)
        sf.write("outputNew.wav", data, samplerate, subtype="PCM_16")

        # Extract text using Speech Recognition
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
            # Read frame from the camera
            rtn, frame = self.cap.read()

            # Detect objects
            class_ids, scores, bboxes = self.model.detect(frame)

            # Draw detected objects if they match the recognized class
            for class_id, score, bbox in zip(class_ids, scores, bboxes):
                x, y, width, height = bbox  # x, y is the left upper corner
                name = self.classes_names[class_id]

                # Look for the recognized class name
                if self.button_clicked and self.command.find(name) > 0:
                    cv2.rectangle(
                        frame, (x, y), (x + width, y + height), (130, 50, 50), 3
                    )
                    cv2.putText(
                        frame,
                        name,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (120, 50, 50),
                        2,
                    )

            # Draw the "Record" button
            cv2.rectangle(
                frame,
                (RecordBoxBoudaries.x_start, RecordBoxBoudaries.y_start),
                (RecordBoxBoudaries.x_end, RecordBoxBoudaries.y_end),
                (153, 0, 0),
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
