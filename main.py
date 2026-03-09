

from app import Application

def main():
    model_path = "hand_detection.keras"
    app = Application(model_path=model_path)
    app.load_model()
    app.run()


if __name__ == "__main__":
    main()
