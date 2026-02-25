# NeuroScope AI — Buildozer Spec for Android APK
# Run: buildozer android debug deploy run

[app]

title = NeuroScope AI
package.name = neuroscope
package.domain = org.neuroscope
source.dir = .
source.include_exts = py,png,jpg,kv,json,ttf

version = 0.1.0

requirements = python3,kivy,opencv,numpy,scipy

orientation = portrait

fullscreen = 1

# Android permissions
android.permissions = CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE

# Android API
android.api = 33
android.minapi = 24
android.ndk = 25b

# Architecture
android.archs = arm64-v8a

# Presplash and icon (placeholders)
# presplash.filename = %(source.dir)s/assets/presplash.png
# icon.filename = %(source.dir)s/assets/icon.png

[buildozer]
log_level = 2
warn_on_root = 1
