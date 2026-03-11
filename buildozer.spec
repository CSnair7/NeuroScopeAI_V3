# NeuroScope AI — Buildozer Spec for Android APK
# Run: buildozer android debug deploy run

[app]

title = NeuroScope AI
package.name = neuroscope
package.domain = org.neuroscope
source.dir = .
source.include_exts = py,png,jpg,kv,json,ttf,onnx,data
source.exclude_dirs = venv,.buildozer,.git,__pycache__,bin,scripts,.github
source.exclude_patterns = *.log,*.pyc,*.pyo

version = 0.1.0

requirements = python3,kivy,opencv,numpy,onnxruntime,android

orientation = portrait

fullscreen = 1

# Allow development mode
p4a.branch = develop

# Android permissions
android.permissions = CAMERA,INTERNET,READ_MEDIA_IMAGES,READ_MEDIA_VIDEO

# Android API
android.api = 35
android.minapi = 24
android.ndk = 28c

# SDK / NDK paths (WSL native filesystem for performance)
android.sdk_path = /home/chinmaynair008/.buildozer/android/platform/android-sdk
android.ndk_path = /home/chinmaynair008/.buildozer/android/platform/android-ndk-r28c
android.accept_sdk_license = True
android.enable_androidx = True

# Architecture
android.archs = arm64-v8a

# Presplash and icon (placeholders)
# presplash.filename = %(source.dir)s/assets/presplash.png
# icon.filename = %(source.dir)s/assets/icon.png

[buildozer]
log_level = 2
warn_on_root = 1
