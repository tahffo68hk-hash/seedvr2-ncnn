package com.seededge.runtime

import org.json.JSONObject

/**
 * Pure helpers for rendering the native pipeline result summary so the
 * error rendering can be unit-tested without an Activity.
 */
object ResultSummary {
    /**
     * Returns "" when the native pipeline reports no error; otherwise
     * "stage=.. code=.. msg=..".  JSONObject.optString turns a JSON null into
     * the literal string "null", so error_stage:null must be treated as "no
     * error" and must never produce a bogus error line.
     */
    fun errorLine(pipeline: JSONObject?): String {
        if (pipeline == null || pipeline.isNull("error_stage")) return ""
        val stage = pipeline.optString("error_stage", "")
        if (stage.isEmpty()) return ""
        val code = pipeline.optString("error_code", "")
        val message = pipeline.optString("error_message", "")
        return "stage=$stage code=$code msg=$message"
    }
}
