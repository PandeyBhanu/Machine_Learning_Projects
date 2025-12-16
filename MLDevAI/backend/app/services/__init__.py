from flask import Flask


def create_app():
    from app.routes import main

    app = Flask(__name__)
    app.config["UPLOAD_FOLDER"] = "uploads"

    app.register_blueprint(main)

    return app
