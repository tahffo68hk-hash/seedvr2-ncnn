package com.seededge.runtime

object SafeResultText {
    const val MAX_CHARS = 16_000
    fun normalize(value: String?): String = value.orEmpty().replace("\u0000", "").takeLast(MAX_CHARS)
    fun replace(previous: String, value: String?): String = normalize(value)
    fun append(previous: String, value: String?): String = normalize(previous + normalize(value) + "\n")
}
