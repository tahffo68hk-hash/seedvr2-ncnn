@rem Minimal Gradle wrapper launcher. The wrapper jar is version-controlled.
@echo off
setlocal
set APP_HOME=%~dp0
if not defined JAVA_HOME set JAVA_HOME=D:\Android\jdk
"%JAVA_HOME%\bin\java.exe" -classpath "%APP_HOME%gradle\wrapper\gradle-wrapper.jar" org.gradle.wrapper.GradleWrapperMain %*
exit /b %ERRORLEVEL%
