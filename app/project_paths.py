"""Folder layout: program modules live in app/, settings and local data in the project folder above."""
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Settings (_ui_settings.json with the API key), run configs, curator_env and
# downloaded models stay in the project folder, so moving the modules into
# app/ keeps every existing local file where it was.
PROJECT_ROOT = os.path.dirname(APP_DIR)
