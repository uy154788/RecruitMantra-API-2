from flask import Flask
from audio_emotion import audio_emotion_bp
from video_text_similarity import video_text_similarity_bp

app = Flask(__name__)

# Register Blueprints
app.register_blueprint(audio_emotion_bp, url_prefix='/audio_emotion')
app.register_blueprint(video_text_similarity_bp, url_prefix='/video_text')

if __name__ == '__main__':
    app.run(debug=False)
