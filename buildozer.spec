[app]

title = Toki AI
package.name = tokiai
package.domain = org.example

source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 0.1

requirements = python3,kivy,pyjnius,android,numpy,aidl

orientation = portrait
osx.python_version = 3
osx.kivy_version = 2.1.0

fullscreen = 0

[buildozer]

log_level = 2
warn_on_root = 1
android.accept_sdk_license = True
