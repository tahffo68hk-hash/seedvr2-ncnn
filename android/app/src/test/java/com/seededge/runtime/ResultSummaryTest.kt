package com.seededge.runtime

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test

class ResultSummaryTest {
    private fun pipeline(json: String): JSONObject = JSONObject(json)

    @Test fun nullErrorStageProducesNoErrorLine() {
        val p = pipeline("""{"error_stage":null,"error_code":null,"error_message":null}""")
        assertEquals("", ResultSummary.errorLine(p))
    }

    @Test fun missingErrorFieldsProducesNoErrorLine() {
        assertEquals("", ResultSummary.errorLine(pipeline("{}")))
        assertEquals("", ResultSummary.errorLine(pipeline("""{"vae_encoder_executed":true}""")))
    }

    @Test fun nullPipelineProducesNoErrorLine() {
        assertEquals("", ResultSummary.errorLine(null))
    }

    @Test fun emptyErrorStageProducesNoErrorLine() {
        assertEquals("", ResultSummary.errorLine(pipeline("""{"error_stage":""}""")))
    }

    @Test fun realErrorIsRendered() {
        val p = pipeline("""{"error_stage":"vae_encoder","error_code":"-100","error_message":"oom"}""")
        assertEquals("stage=vae_encoder code=-100 msg=oom", ResultSummary.errorLine(p))
    }
}
