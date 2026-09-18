from flask import Flask, render_template


def create_app():
    app = Flask(__name__)
    app.secret_key = "csc8830-cv-dev-key"  # fine for a local class demo, not for production

    from homeworks.hw2_calibration.routes import bp as hw2_bp
    from homeworks.hw3_blurring.routes import bp as hw3_bp
    from homeworks.hw4_segmentation.routes import bp as hw4_bp

    app.register_blueprint(hw2_bp, url_prefix="/hw2")
    app.register_blueprint(hw3_bp, url_prefix="/hw3")
    app.register_blueprint(hw4_bp, url_prefix="/hw4")

    @app.route("/")
    def landing():
        return render_template("landing.html")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
