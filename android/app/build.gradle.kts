plugins { id("com.android.application"); id("org.jetbrains.kotlin.android") }

android {
    namespace = "com.seededge.runtime"
    compileSdk = 35
    ndkVersion = "26.3.11579264"
    defaultConfig {
        applicationId = "com.seededge.runtime"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "18A.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk { abiFilters += listOf("arm64-v8a") }
        externalNativeBuild { cmake { arguments += listOf("-DANDROID_STL=c++_static", "-DNCNN_VULKAN=ON", "-DNCNN_OPENMP=OFF"); cppFlags += listOf("-std=c++17", "-fexceptions", "-fno-rtti") } }
    }
    externalNativeBuild { cmake { path = file("../CMakeLists.txt"); cmake.version = "3.22.1" } }
    buildTypes {
        debug { isMinifyEnabled = false }
        release { isMinifyEnabled = false; proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro") }
    }
    packaging { jniLibs { useLegacyPackaging = false } }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}
